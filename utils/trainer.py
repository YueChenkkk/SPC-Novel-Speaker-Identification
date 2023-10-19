
import os
import time
import torch
import torch.distributed as dist


def gather_tensors(to_be_gathered):
    gathered = [torch.zeros_like(to_be_gathered) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, to_be_gathered)
    gathered = torch.cat(gathered, dim=0)
    return gathered


def gather_objects(to_be_gathered):
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, to_be_gathered)
    gathered = [x.to(gathered[0].device) for x in gathered]
    gathered = torch.cat(gathered, dim=0)
    return gathered


class DDPTrainer:
    def __init__(
        self, 
        model,
        optimizer, 
        lr_scheduler,
        current_epoch,
        device
    ):
        """This is the class that takes care of distributed data parallel training.
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.device = device

        self.epoch_counter = current_epoch

    def run_epoch(self, data_loader, is_train, args):
        """
        Train the model for one epoch on the data of train_loader and return loggings 
        """
        if is_train:
            self.epoch_counter += 1
            self.model.train()
            self.optimizer.zero_grad()
            if self.epoch_counter > 0:
                self.lr_scheduler.step()
        else:
            self.model.eval()
        
        time_start = time.time()
        total_loss = base_loss = neighbour_loss = role_loss = 0
        backward_counter = 0
        base_pred_labels = []
        base_gold_labels = []
        neighbour_utter_pred_labels = []
        neighbour_utter_gold_labels = []
        role_pred_labels = []
        role_gold_labels = []
        for forward_inputs, other_infos in data_loader:
            bert_inputs = forward_inputs['bert_inputs']
            bert_inputs = {k: v.to(self.device) for k, v in bert_inputs.items()}
            forward_inputs['bert_inputs'] = bert_inputs
            forward_inputs['args'] = args

            label_idx = forward_inputs['label_idx']
            neighbour_utter_label_idxes = forward_inputs['neighbour_utter_label_idxes']
            narr_role_label_idxes = forward_inputs['narr_role_label_idxes']

            if is_train:
                backward_counter += 1
                if backward_counter % args.update_freq != 0:
                    with self.model.no_sync():
                        outputs = self.model(**forward_inputs)
                        outputs.total_loss.backward()
                else:
                    outputs = self.model(**forward_inputs)
                    outputs.total_loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            else:
                with torch.no_grad(), self.model.no_sync():
                    outputs = self.model(**forward_inputs)

            total_loss += outputs.total_loss.item()
            base_loss += outputs.base_loss.item()
            neighbour_loss += outputs.neighbour_speaker_loss.item()
            role_loss += outputs.role_prediction_loss.item()

            base_gold_labels.extend(label_idx)
            base_pred_labels.extend([torch.argmax(x).item() for x in outputs.base_scores])
            neighbour_utter_gold_labels.extend([y for x in neighbour_utter_label_idxes for y in x]) 
            neighbour_utter_pred_labels.extend([torch.argmax(x).item() for x in outputs.neighbour_speaker_prediction_scores])
            role_gold_labels.extend([y for x in narr_role_label_idxes for y in x]) 
            role_pred_labels.extend([torch.argmax(x).item() for x in outputs.role_prediction_scores])

        time_end = time.time()
        epoch_time = time_end - time_start

        total_loss /= len(data_loader)
        base_loss /= len(data_loader)
        neighbour_loss /= len(data_loader)
        role_loss /= len(data_loader)
        base_gold_labels = torch.tensor(base_gold_labels).long().to(self.device)
        base_pred_labels = torch.tensor(base_pred_labels).long().to(self.device)
        neighbour_utter_gold_labels = torch.tensor(neighbour_utter_gold_labels).long().to(self.device)
        neighbour_utter_pred_labels = torch.tensor(neighbour_utter_pred_labels).long().to(self.device)
        role_gold_labels = torch.tensor(role_gold_labels).long().to(self.device)
        role_pred_labels = torch.tensor(role_pred_labels).long().to(self.device)

        # gather results from all processes
        base_gathered_gold = gather_tensors(base_gold_labels)
        base_gathered_pred = gather_tensors(base_pred_labels)
        neighbour_utter_gathered_gold = gather_objects(neighbour_utter_gold_labels).to(base_gathered_gold.device)
        neighbour_utter_gathered_pred = gather_objects(neighbour_utter_pred_labels).to(base_gathered_gold.device)
        role_gathered_gold = gather_objects(role_gold_labels).to(base_gathered_gold.device)
        role_gathered_pred = gather_objects(role_pred_labels).to(base_gathered_gold.device)
        base_n_correct = torch.sum(base_gathered_gold == base_gathered_pred).item()
        base_accuracy = base_n_correct / base_gathered_gold.size(0)
        neighbour_utter_n_correct = torch.sum(neighbour_utter_gathered_gold == neighbour_utter_gathered_pred).item()
        role_n_correct = torch.sum(role_gathered_gold == role_gathered_pred).item()
        
        try:
            neighbour_utter_accuracy = neighbour_utter_n_correct / neighbour_utter_gathered_gold.size(0)
        except ZeroDivisionError:
            neighbour_utter_accuracy = 0.0
            
        try:
            role_accuracy = role_n_correct / role_gathered_gold.size(0)
        except ZeroDivisionError:
            role_accuracy = 0.0

        log_outputs = {
            'total_loss': total_loss,
            'base_loss': base_loss,
            'neighbour_loss': neighbour_loss,
            'role_loss': role_loss,
            'accuracy': base_accuracy,
            'neighbour_acc': neighbour_utter_accuracy,
            'role_acc': role_accuracy,
            'time': epoch_time
        }
        return log_outputs

    def valid_epoch(self, valid_loader):
        """
        Validate on the data of valid_loader
        """
        self.model.eval()
        
        time_start = time.time()
        
        total_loss = base_loss = neighbour_loss = role_loss = 0
        base_pred_labels = []
        base_gold_labels = []
        neighbour_utter_pred_labels = []
        neighbour_utter_gold_labels = []
        role_pred_labels = []
        role_gold_labels = []
        for forward_inputs, other_infos in valid_loader:
            bert_inputs = forward_inputs['bert_inputs']
            bert_inputs = {k: v.to(self.device) for k, v in bert_inputs.items()}
            forward_inputs['bert_inputs'] = bert_inputs

            label_idx = forward_inputs['label_idx']
            neighbour_utter_label_idxes = forward_inputs['neighbour_utter_label_idxes']
            narr_role_label_idxes = forward_inputs['narr_role_label_idxes']

            with torch.no_grad():
                outputs = self.model(**forward_inputs)

            total_loss += outputs.total_loss.item()
            base_loss += outputs.base_loss.item()
            neighbour_loss += outputs.neighbour_speaker_loss.item()
            role_loss += outputs.role_prediction_loss.item()

            base_gold_labels.extend(label_idx)
            base_pred_labels.extend([torch.argmax(x).item() for x in outputs.base_scores])
            neighbour_utter_gold_labels.extend([y for x in neighbour_utter_label_idxes for y in x]) 
            neighbour_utter_pred_labels.extend([torch.argmax(x).item() for x in outputs.neighbour_speaker_prediction_scores])
            role_gold_labels.extend([y for x in narr_role_label_idxes for y in x]) 
            role_pred_labels.extend([torch.argmax(x).item() for x in outputs.role_prediction_scores])

        time_end = time.time()
        epoch_time = time_end - time_start

        total_loss /= len(valid_loader)
        base_loss /= len(valid_loader)
        neighbour_loss /= len(valid_loader)
        role_loss /= len(valid_loader)
        base_gold_labels = torch.tensor(base_gold_labels).long().to(self.device)
        base_pred_labels = torch.tensor(base_pred_labels).long().to(self.device)
        neighbour_utter_gold_labels = torch.tensor(neighbour_utter_gold_labels).long().to(self.device)
        neighbour_utter_pred_labels = torch.tensor(neighbour_utter_pred_labels).long().to(self.device)
        role_gold_labels = torch.tensor(role_gold_labels).long().to(self.device)
        role_pred_labels = torch.tensor(role_pred_labels).long().to(self.device)

        # gather results from all processes
        base_gathered_gold = gather_tensors(base_gold_labels)
        base_gathered_pred = gather_tensors(base_pred_labels)
        neighbour_utter_gathered_gold = gather_objects(neighbour_utter_gold_labels)
        neighbour_utter_gathered_pred = gather_objects(neighbour_utter_pred_labels)
        role_gathered_gold = gather_objects(role_gold_labels)
        role_gathered_pred = gather_objects(role_pred_labels)

        base_n_correct = torch.sum(base_gathered_gold == base_gathered_pred).item()
        base_accuracy = base_n_correct / base_gathered_gold.size(0)
        neighbour_utter_n_correct = torch.sum(neighbour_utter_gathered_gold == neighbour_utter_gathered_pred).item()
        role_n_correct = torch.sum(role_gathered_gold == role_gathered_pred).item()
        
        try:
            neighbour_utter_accuracy = neighbour_utter_n_correct / neighbour_utter_gathered_gold.size(0)
        except ZeroDivisionError:
            neighbour_utter_accuracy = 0.0
        
        try:
            role_accuracy = role_n_correct / role_gathered_gold.size(0)
        except ZeroDivisionError:
            role_accuracy = 0.0

        log_outputs = {
            'total_loss': total_loss,
            'base_loss': base_loss,
            'neighbour_loss': neighbour_loss,
            'role_loss': role_loss,
            'accuracy': base_accuracy,
            'neighbour_acc': neighbour_utter_accuracy,
            'role_acc': role_accuracy,
            'time': epoch_time
        }
        return log_outputs

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def update(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


