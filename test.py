# Training script

import os
import sys
import json
import copy
import time
import logging
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist

from transformers import BertTokenizer

from utils.load_data import get_dataloader
from models.model_mlm import SindBertMaskedLM
from utils.eval_utils import *
from utils.file_utils import *

parser = ArgumentParser()
parser.add_argument("--world-size", type=int, default=1)
parser.add_argument("--output-name", type=str, default='')
parser.add_argument("--ckpt-dir", type=str, default='')
parser.add_argument("--data-dir", type=str, default='')
parser.add_argument("--batch-size", type=int, default=4)
args = parser.parse_args()

with open(os.path.join(args.ckpt_dir, 'args.json'), encoding='utf-8') as fin:
    arguments = json.load(fin)

for k, v in arguments.items():
    if k not in args.__dict__:
        args.__dict__[k] = v

# logging
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    '%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()
logger.setLevel('DEBUG')


def test(local_rank):
    rank = local_rank

    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    
    # load tokenizer
    print(args.ckpt_dir)
    tokenizer = BertTokenizer.from_pretrained(args.ckpt_dir)

    # get data loader
    test_loader = get_dataloader(split_name='test', tokenizer=tokenizer, args=args)
    test_neighbour_num = sum(len(x) for x in test_loader.dataset.data_dict['neighbour_utter_mask_poses'])
    test_role_pred_num = sum(len(x) for x in test_loader.dataset.data_dict['role_mask_poses'])

    if rank == 0:
        test_log_dir = os.path.join(args.ckpt_dir, 'test_log', args.output_name)

        if not os.path.exists(test_log_dir):
            os.makedirs(test_log_dir)

        # logging
        logging_path = os.path.join(test_log_dir, 'test_log.log')
        fh = logging.FileHandler(logging_path, 'w', encoding='utf-8')      
        ch = logging.StreamHandler()
        fh.setLevel('INFO')
        ch.setLevel('INFO')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

        logger.info(f"Number of testing utterances: {len(test_loader.dataset)}")
        logger.info(f"Number of testing neighbour utterances: {test_neighbour_num}")
        logger.info(f"Number of testing role predictions: {test_role_pred_num}")

    logger.info("Loading the checkpoint ...")

    # initialize the model
    model = SindBertMaskedLM.from_pretrained(args.ckpt_dir)
    model = model.to(device)
    model.eval()

    time_start = time.time()

    test_loss = 0
    test_preds = {}
    test_targets = {}
    pred_res_dict = {}
    neighbour_utter_pred_labels = []
    neighbour_utter_gold_labels = []
    for forward_inputs, other_infos in tqdm(test_loader):
        bert_inputs = forward_inputs['bert_inputs']
        bert_inputs = {k: v.to(device) for k, v in bert_inputs.items()}
        forward_inputs['bert_inputs'] = bert_inputs
        forward_inputs['args'] = args

        neighbour_utter_label_idxes = forward_inputs['neighbour_utter_label_idxes']

        batch_idx2roleid = other_infos['idx2roleid']
        instance_ids = other_infos['instance_id']
        batch_label_idx = forward_inputs['label_idx']

        with torch.no_grad():
            outputs = model(**forward_inputs)

        neighbour_utter_gold_labels.extend([y for x in neighbour_utter_label_idxes for y in x]) 
        neighbour_utter_pred_labels.extend([torch.argmax(x).item() for x in outputs.neighbour_speaker_prediction_scores])

        test_loss += outputs.total_loss.item()
        for inst_idx in range(len(instance_ids)):
            inst_scores = outputs.base_scores[inst_idx]
            idx2roleid = batch_idx2roleid[inst_idx]
            instance_id = instance_ids[inst_idx]
            label_idx = batch_label_idx[inst_idx]
            distance = other_infos['distance'][inst_idx]
            
            role_ids = [idx2roleid[i] for i in range(inst_scores.size(0))]
            speaker_id = idx2roleid[label_idx]
            
            update_pred(
                pred_dict=test_preds, 
                logits=torch.nn.Tanh()(inst_scores), 
                instance_ids=[instance_id] * inst_scores.size(0),
                role_ids=role_ids
            )

            test_targets.update({instance_id: speaker_id})

            pred_res_dict.update({
                instance_id: {
                    'distance': distance,
                    'gold_id': speaker_id,
                    'pred_id': idx2roleid[torch.argmax(inst_scores, dim=0).item()]

                }
            })

    time_end = time.time()
    test_time = time_end - time_start
    logger.info(f"Testing time cost: {test_time}")
    
    test_loss /= len(test_loader)

    gathered_test_preds = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_test_preds, test_preds)
    sum_test_preds = summarize_preds(gathered_test_preds)

    gathered_targets = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_targets, test_targets)
    sum_test_targets = {k: v for t_dict in gathered_targets for k, v in t_dict.items()}

    def gather_objects(to_be_gathered):
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, to_be_gathered)
        gathered = torch.cat([x.to(gathered[0].device) for x in gathered], dim=0)
        return gathered

    neighbour_utter_gold_labels = torch.tensor(neighbour_utter_gold_labels).long().to(device)
    neighbour_utter_pred_labels = torch.tensor(neighbour_utter_pred_labels).long().to(device)
    neighbour_gathered_gold = gather_objects(neighbour_utter_gold_labels)
    neighbour_gathered_pred = gather_objects(neighbour_utter_pred_labels)
    neighbour_n_correct = torch.sum(neighbour_gathered_gold == neighbour_gathered_pred).item()
    try:
        neighbour_accuracy = neighbour_n_correct / neighbour_gathered_gold.size(0)
    except ZeroDivisionError:
        neighbour_accuracy = 0.0

    gathered_res_dict = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_res_dict, pred_res_dict)
    sum_res_dict = {k: v for r_dict in gathered_res_dict for k, v in r_dict.items()}
    distance_exp_output_list = [{'correct': 0, 'wrong': 0} for i in range(11)]
    for inst_id, inst_dict in sum_res_dict.items():
        distance = inst_dict['distance']
        pred_spk_id = inst_dict['pred_id']
        gold_spk_id = inst_dict['gold_id']

        if pred_spk_id == gold_spk_id:
            distance_exp_output_list[distance]['correct'] += 1
        else:
            distance_exp_output_list[distance]['wrong'] += 1

    if rank == 0:
        test_acc, test_num = compute_acc(pred_dict=sum_test_preds, target_dict=sum_test_targets)

        test_info = args.__dict__
        test_info.update({
            'test_loss': test_loss,
            'test_acc': test_acc,
            'aux_acc': neighbour_accuracy,
            'test_num': test_num
        })
        with open(os.path.join(test_log_dir, 'test_info.json'), 'w', encoding='utf-8') as fout:
            json.dump(test_info, fout, indent=4)

        with open(os.path.join(test_log_dir, 'accuracies.json'), 'w', encoding='utf-8') as fout:
            json.dump(test_acc, fout, indent=4)

        with open(os.path.join(test_log_dir, 'acc_by_dist.json'), 'w', encoding='utf-8') as fout:
            json.dump(distance_exp_output_list, fout, indent=4)


if __name__ == '__main__':
    mp.spawn(test, nprocs=args.world_size, join=True)

