

import json
import os
import pickle
import re
import random
from typing import Dict, List, Tuple
import torch
import numpy as np
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from transformers import TokenClassificationPipeline


def _load_instances(instance_file):
    if not os.path.exists(instance_file):
        return None
    # load instance information
    with open(instance_file, encoding='utf-8') as fin:
        instance_dict = json.load(fin)
    return instance_dict


def mask_role_mentions(line_tokens, can_mask_cidxs, prob):
    masked_tokens = []
    mask_poses = []
    mask_labels = []

    for token in line_tokens:
        # find a role mention
        if token in can_mask_cidxs:
            # mask it by a give probability
            if random.uniform(0, 1) < prob:
                mask_poses.append(len(masked_tokens))
                mask_labels.append(token)
                masked_tokens.append('[MASK]')
            else:
                masked_tokens.append(token)
        else:
            masked_tokens.append(token)

    return {
        'tokens_in_line': masked_tokens,
        'line_role_mask_poses': mask_poses,
        'line_role_mask_labels': mask_labels
    }


def compute_distance(roleid2poses, speaker_id, dialogue_idx):
    if speaker_id in roleid2poses:
        poses = roleid2poses[speaker_id]
        pos_dists = [np.abs(pos[0] - dialogue_idx) for pos in poses]
        distance = min(pos_dists)
    else:
        distance = 10
    return distance


def collect_samples_from_file(
    instance_file, 
    tokenizer,
    stage,
    role_mask_prob,
    skip_only_one_candidate,
    args
):
    """Load training samples from a book
    """
    samples = {
        'instance_id': [],
        'input_ids': [],
        'idx2roleid': [],
        'target_token_ids': [],
        'mask_pos': [],
        'label_idx': [],
        'neighbour_utter_mask_poses': [],
        'neighbour_utter_label_idxes': [],
        'role_mask_poses': [],
        'role_label_idxes': [],
        'distance': []
    }

    instance_dict = _load_instances(instance_file)
    if not instance_dict:
        return samples

    for instance_id, instance in instance_dict.items():
        roleid2cidx = instance['roleid2idx']
        text_lines = instance['text']
        target_idx = instance['target_idx']
        dialogue_idxes = instance['dialogue_idxes']
        speaker_ids = instance['speaker_ids']
        roleid2poses = instance['roleid2poses']

        if skip_only_one_candidate and len(roleid2cidx) == 1:
            continue
        
        dialogue_idx2speaker_id = dict(zip(dialogue_idxes, speaker_ids))

        # select the auxiliary dialogues to involve
        meta_idx = dialogue_idxes.index(target_idx)
        neighbour_utter_idxes = []
        neighbour_utter_spk_ids = []
        for idx in dialogue_idxes[meta_idx - args.left_aux:meta_idx + args.right_aux + 1]:
            if idx == target_idx:
                continue

            speaker_id = dialogue_idx2speaker_id.get(idx, None)
            if speaker_id in roleid2cidx:
                neighbour_utter_idxes.append(idx)
                neighbour_utter_spk_ids.append(speaker_id)

        target_token_ids = tokenizer.convert_tokens_to_ids([f'[C{idx}]' for idx in range(len(roleid2cidx))])
        label_cidx = roleid2cidx.get(dialogue_idx2speaker_id[target_idx], '[C0]')
        cidx2roleid = {cidx: role_id for role_id, cidx in roleid2cidx.items()}
        idx2roleid = {idx: cidx2roleid[f'[C{idx}]'] for idx in range(len(roleid2cidx))}
        
        cidx2idx = lambda x: int(re.search(r'[0-9]+', x).group(0))
        label_idx = cidx2idx(label_cidx)
        neighbour_utter_label_idxes = [cidx2idx(roleid2cidx[x]) for x in neighbour_utter_spk_ids]

        # generate prompt template
        mask_token = '[MASK]'
        # prompt sequence 2
        # prompt sequence 2
        if args.prompt_type == '2':
            prompt_left = ''
            prompt_right = ''

        # # prompt sequence 3
        elif args.prompt_type == '3':
            prompt_left = '（'
            prompt_right = '说了这句话）'

        # prompt sequence 4
        elif args.prompt_type == '4':
            prompt_left = '（'
            prompt_right = '是一个友善的人）'

        # prompt sequence 5
        elif args.prompt_type == '5':
            prompt_left = '（很显然，'
            prompt_right = '说了这句话）'

        # prompt sequence 6
        elif args.prompt_type == '6':
            prompt_left = '（'
            prompt_right = '说）'

        # prompt sequence 7
        elif args.prompt_type == '7':
            prompt_left = '（'
            prompt_right = '）'

        # prompt sequence 8
        elif args.prompt_type == '8':
            prompt_left = ''
            prompt_right = '说了这句话'

        # # prompt sequence 1
        else:
            prompt_left = '（这句话是'
            prompt_right = '说的）'

        prompt_left_len = len(prompt_left)
        prompt_tokens = list(prompt_left) + [mask_token] + list(prompt_right)

        # generate the masked input tokens
        # first determine a group of symbols (cidx) that can be masked,
        # only mask the one mentioned in sentences other than the preceding
        # and the following sentences.
        can_mask_cidxs = []
        for role_id, poses in roleid2poses.items():
            cidx = roleid2cidx[role_id]
            for pos in poses:
                if pos[0] not in [target_idx - 1, target_idx + 1]:
                    can_mask_cidxs.append(cidx)
                    break

        main_mask_pos = -1
        role_mask_poses = []
        role_mask_labels = []
        neighbour_utter_mask_poses = []
        token_list = ['[CLS]']
        for idx, line in enumerate(text_lines):
            tokens_in_line = tokenizer.tokenize(line)

            # mask explicit narrative evidence in the preceding and the following sentences
            if idx in [target_idx - 1, target_idx + 1] and stage == 'train' and role_mask_prob > 0.0:
                mask_res = mask_role_mentions(tokens_in_line, can_mask_cidxs=can_mask_cidxs, prob=role_mask_prob)
                tokens_in_line = mask_res['tokens_in_line']
                role_mask_poses.extend([len(token_list) + x for x in mask_res['line_role_mask_poses']])
                role_mask_labels.extend([cidx2idx(x) for x in mask_res['line_role_mask_labels']])

            token_list += tokens_in_line
            
            if idx == target_idx:
                main_mask_pos = len(token_list) + prompt_left_len
                token_list += prompt_tokens
            elif idx in neighbour_utter_idxes:
                neighbour_utter_mask_poses.append(len(token_list) + prompt_left_len)
                token_list += prompt_tokens

        token_list += ['[SEP]']

        assert len(neighbour_utter_mask_poses) == len(neighbour_utter_label_idxes)

        speaker_id = dialogue_idx2speaker_id[target_idx]
        distance = compute_distance(roleid2poses, speaker_id, target_idx)

        samples['instance_id'].append(instance_id)
        samples['input_ids'].append(tokenizer.convert_tokens_to_ids(token_list))
        samples['idx2roleid'].append(idx2roleid)
        samples['target_token_ids'].append(target_token_ids)
        samples['mask_pos'].append(main_mask_pos)
        samples['label_idx'].append(label_idx)
        samples['neighbour_utter_mask_poses'].append(neighbour_utter_mask_poses)
        samples['neighbour_utter_label_idxes'].append(neighbour_utter_label_idxes)
        samples['role_mask_poses'].append(role_mask_poses)
        samples['role_label_idxes'].append(role_mask_labels)
        samples['distance'].append(distance)

    return samples


def get_samples(split_name, tokenizer, role_mask_prob, skip_only_one_candidate, args):
    assert split_name in ['train', 'dev', 'test'], f"split_name should be one of ['train', 'dev', 'test'], but got {split_name}"

    instance_file = os.path.join(args.data_dir, f'{split_name}/{split_name}_instances.json')
    return collect_samples_from_file(instance_file, tokenizer, split_name, role_mask_prob, skip_only_one_candidate, args)

