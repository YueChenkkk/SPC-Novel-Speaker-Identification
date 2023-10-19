# Training script

import os
from argparse import ArgumentParser

import json
import time
from tqdm import tqdm
import logging

import numpy as np
import torch

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import BertConfig, BertTokenizer

from utils.load_data import get_dataloader
from models.model_mlm import SindBertMaskedLM
from utils.trainer import *
from utils.checkpoint_utils import *

parser = ArgumentParser()

# loading arguments from the environment variables
parser.add_argument("--world-size", type=int, default=1)

parser.add_argument("--data-dir", type=str, default="")

parser.add_argument("--epoch-num", type=int, default=50)
parser.add_argument("--total-batch-size", type=int, default=128)
parser.add_argument("--batch-size-per-gpu", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--lr-gamma", type=float, default=0.98)
parser.add_argument("--early-stop", type=int, default=5)
parser.add_argument("--margin", type=float, default=1.0)

parser.add_argument("--max-len", type=int, default=512)
parser.add_argument("--left-aux", type=int, default=1)
parser.add_argument("--right-aux", type=int, default=1)
parser.add_argument("--prompt-type", type=str, default='3')

parser.add_argument("--role-mask-prob", type=float, default=0.5)
parser.add_argument("--lbd1", type=float, default=0.3)
parser.add_argument("--lbd2", type=float, default=0.3)

parser.add_argument("--pretrained-dir", type=str, default="")
parser.add_argument("--ckpt-dir", type=str, default="")
parser.add_argument("--do-resume", action='store_true', default=False)

args = parser.parse_args()

args.update_freq = int(args.total_batch_size // (args.world_size * args.batch_size_per_gpu))


# training log
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    '%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()
logger.setLevel('DEBUG')


def train(local_rank):
    """
    Training script
    """

    # Distributed settings
    rank = local_rank

    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    # initialize logger
    if rank == 0:
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

        logging_path = os.path.join(args.ckpt_dir, LOG_FILE)
        if args.do_resume:
            fh = logging.FileHandler(logging_path, 'a', encoding='utf-8')
        else:
            fh = logging.FileHandler(logging_path, 'w', encoding='utf-8')

        ch = logging.StreamHandler()
        fh.setLevel('INFO')
        ch.setLevel('DEBUG')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    # Initialize model and tokenizer
    load_dir = args.ckpt_dir if args.do_resume else args.pretrained_dir
    
    config = BertConfig.from_pretrained(load_dir)
    tokenizer = BertTokenizer.from_pretrained(load_dir)
    tokenizer.add_special_tokens({'additional_special_tokens': [f'[C{i}]' for i in range(50)]})
    
    model = SindBertMaskedLM.from_pretrained(load_dir, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    ddp_model = DDP(model, device_ids=[device], find_unused_parameters=False)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr / args.update_freq)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)

    start_epoch = -1
    best_valid_acc = 0
    patience_counter = 0

    if args.do_resume:
        try:
            train_state = load_train_state(args.ckpt_dir, device)
            resume_training_state(
                train_state,
                optimizer,
                lr_scheduler
            )
            start_epoch = train_state['epoch']
            best_valid_acc = train_state['best_valid_acc']
            patience_counter = train_state['patience_counter']

            if rank == 0:
                logger.info(f"Resumed training from checkpoint directory {args.ckpt_dir}.")
        except OSError:
            logger.info(
                f"""
                No saved checkpoint can be found in directory
                {args.ckpt_dir}.
                Just train from scratch.
                """
            )
    elif rank == 0:
        logger.info(f"Train from scratch.")

    # announce the current training state
    if rank == 0: 
        logger.info(
            f"""
            Currently,
            best validation accuracy: {best_valid_acc},
            patience counter: {patience_counter},
            learning rate: {lr_scheduler.get_last_lr()},
            world size: {args.world_size},
            batch size per GPU: {args.batch_size_per_gpu},
            update frequency: {args.update_freq},
            actual total batch size: {args.world_size * args.batch_size_per_gpu * args.update_freq}.
            """
        )

    if rank == 0:
        logger.info(f"WORLD_SIZE={args.world_size}")
        logger.info("Initailizing trainer...")
    
    trainer = DDPTrainer(
        model=ddp_model,
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler,
        current_epoch=start_epoch,
        device=device
    )

    # build data loader
    logger.info("Preparing data...")
    train_loader = get_dataloader(split_name='train', tokenizer=tokenizer, args=args)
    valid_loader = get_dataloader(split_name='dev', tokenizer=tokenizer, args=args)

    train_neighbour_num = sum(len(x) for x in train_loader.dataset.data_dict['neighbour_utter_mask_poses'])
    valid_neighbour_num = sum(len(x) for x in valid_loader.dataset.data_dict['neighbour_utter_mask_poses'])
    train_role_pred_num = sum(len(x) for x in train_loader.dataset.data_dict['role_mask_poses'])
    valid_role_pred_num = sum(len(x) for x in valid_loader.dataset.data_dict['role_mask_poses'])

    if rank == 0:
        logger.info(f"Number of training utterances: {len(train_loader.dataset)}")
        logger.info(f"Number of training neighbour utterances: {train_neighbour_num}")
        logger.info(f"Number of training role predictions: {train_role_pred_num}")
        logger.info(f"Number of validating utterances: {len(valid_loader.dataset)}")
        logger.info(f"Number of validating neighbour utterances: {valid_neighbour_num}")
        logger.info(f"Number of validating role predictions: {valid_role_pred_num}")

        # print example
        logger.debug("############## EXAMPLE ###############")
        train_test_iter = iter(train_loader)
        feature, info = train_test_iter.__next__()
        bert_inputs = feature['bert_inputs']
        logger.debug("BERT INPUTS:")
        logger.debug(repr(bert_inputs))
        logger.debug("TEXT EXAMPLE:")
        for i in range(len(bert_inputs['input_ids'])):
            logger.debug(''.join(tokenizer.convert_ids_to_tokens(bert_inputs['input_ids'][i])))
        logger.debug("TARGET LABEL:")
        logger.debug(feature['label_idx'])
        logger.debug("NEIGHBOUR SPEAKER LABELS:")
        logger.debug(feature['neighbour_utter_label_idxes'])
        logger.debug("ROLE PREDICTION LABELS:")
        logger.debug(feature['narr_role_label_idxes'])
        logger.debug("INFO:")
        logger.debug(info)

        # print example
        logger.debug("############## EXAMPLE ###############")
        valid_test_iter = iter(valid_loader)
        feature, info = valid_test_iter.__next__()
        bert_inputs = feature['bert_inputs']
        logger.debug("BERT INPUTS:")
        logger.debug(repr(bert_inputs))
        logger.debug("TEXT EXAMPLE:")
        for i in range(len(bert_inputs['input_ids'])):
            logger.debug(''.join(tokenizer.convert_ids_to_tokens(bert_inputs['input_ids'][i])))
        logger.debug("TARGET LABEL:")
        logger.debug(feature['label_idx'])
        logger.debug("NEIGHBOUR SPEAKER LABELS:")
        logger.debug(feature['neighbour_utter_label_idxes'])
        logger.debug("ROLE PREDICTION LABELS:")
        logger.debug(feature['narr_role_label_idxes'])
        logger.debug("INFO:")
        logger.debug(info)

    # training loop
    if rank == 0:
        logger.info("Training Begins...")

    for epoch in range(start_epoch + 1, args.epoch_num):
        if rank == 0:
            logger.info('Epoch: %d' % (epoch + 1))
            
        train_log_output = trainer.run_epoch(train_loader, is_train=True, args=args)

        # logging
        if rank == 0:
            logger.info(f'train--{train_log_output}')

        # Validation
        valid_log_output = trainer.run_epoch(valid_loader, is_train=False, args=args)

        # logging
        if rank == 0:
            logger.info(f'valid--{valid_log_output}')

        # save the model with best performance
        if valid_log_output['accuracy'] > best_valid_acc:
            best_valid_acc = valid_log_output['accuracy']
            patience_counter = 0

            if rank == 0:
                logger.info('New best!!!!!!!')
                logger.info('Saving checkpoint...')

                save_checkpoint(
                    args.ckpt_dir,
                    config,
                    tokenizer,
                    model, 
                    optimizer,
                    lr_scheduler,
                    trainer.epoch_counter,
                    patience_counter,
                    {
                        'train_log': train_log_output,
                        'valid_log': valid_log_output,
                        'best_valid_acc': best_valid_acc
                    },
                    args
                )
        else:
            patience_counter += 1

        # early stopping
        if patience_counter > args.early_stop:
            if rank == 0:
                logger.info("Early stopping...")
            break

    if rank == 0:
        logger.removeHandler(fh)
        logger.removeHandler(ch)
        fh.close()


if __name__ == '__main__':
    mp.spawn(train, nprocs=args.world_size, join=True)

