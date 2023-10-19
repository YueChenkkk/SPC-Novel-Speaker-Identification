
import os
import json
import copy
import re
from tqdm import tqdm
from collections import OrderedDict
from argparse import ArgumentParser


def convert_mentions_to_signs(raw_sents, alias2id, max_len):
    '''
    Converts role names to masked names `C[X]`,
    returns the sentences after this conversion and the 
    mappings from role IDs to `C[X]`.
    '''
    tokens = []
    roleid2poses = {}
    roleid2idx = {}

    sorted_alias_id = sorted(alias2id.items(), key=lambda x: -len(x[0]))

    for idx, sent in enumerate(raw_sents):
        new_token_list = []

        i = 0
        while i < len(sent):

            is_name = False
            for name, roleid in sorted_alias_id:
                if sent[i:i + len(name)] == name:
                    roleid = alias2id[name]

                    if roleid not in roleid2poses:
                        roleid2idx[roleid] = f'[C{len(roleid2idx)}]'
                        roleid2poses[roleid] = []
                    roleid2poses[roleid].append((idx, len(new_token_list)))

                    new_token_list.append(roleid2idx[roleid])
                    is_name = True
                    i += len(name)
                    break

            if not is_name:
                new_token_list.append(sent[i])
                i += 1
            
        new_token_list = new_token_list[:max_len]
        tokens.append(new_token_list)

    return tokens, roleid2poses, roleid2idx


def expand_ctx(share_ctx_span, tokens, max_len):
    expand_start, expand_end = share_ctx_span
    start, end = share_ctx_span

    # try to expand the left side first
    for i in range(start - 1, -1, -1):
        if sum(len(x) for x in tokens[i:end]) <= max_len:
            expand_start = i
        else:
            break

    # then expand the right side if possible
    for i in range(end + 1, len(tokens)):
        if sum(len(x) for x in tokens[expand_start:i]) <= max_len:
            expand_end = i
        else:
            break

    return expand_start, expand_end


def build_instance(tokens, roleid2poses, max_len):
    """
    Build an instance
    """

    def dist_func(role_pos, tokens, left_priority):
        '''Distance function of each mention (now characters).
        Compute the character-level distance between the mention and the dialogue.
        '''
        dist = 0

        if left_priority:
            right_degrade = 10000
        else:
            right_degrade = 0

        if role_pos[0] > SSWS:
            dist += sum(len(x) for x in tokens[SSWS + 1:role_pos[0]])
            dist += role_pos[1]
            dist += right_degrade
        elif role_pos[0] < SSWS:
            dist += sum(len(x) for x in tokens[role_pos[0] + 1:SSWS])
            dist += len(tokens[role_pos[0]]) - role_pos[1] - 1
        else:
            dist += DIST_IN_DLG
        
        return dist

    try:
        left_priority = tokens[SSWS - 1][-1] in [':', '：', ',', '，']
    except IndexError:
        left_priority = False

    # nearest mention selection
    roleid2nm = {}
    for roleid, poses in roleid2poses.items():
        pos_dist = [(pos, dist_func(pos, tokens, left_priority=left_priority)) for pos in poses]
        roleid2nm.update({roleid: min(pos_dist, key=lambda x: x[1])})

    ctx_span = (SSWS, SSWS + 1)
    covered_roleid2pos = {}

    sorted_role_poses = sorted(roleid2nm.items(), key=lambda x:x[1][1])
    for idx, (roleid, pos_dist) in enumerate(sorted_role_poses):
        # check whether each candidate speaking role can be covered in the shared context,
        # starting from the nearest one
        pos, _ = pos_dist
        if pos[0] < SSWS:
            attempt_ctx_span = (pos[0], ctx_span[1])
        elif pos[0] > SSWS:
            attempt_ctx_span = (ctx_span[0], pos[0] + 1)
        else:
            attempt_ctx_span = ctx_span

        attempt_len = sum(len(x) for x in tokens[attempt_ctx_span[0]:attempt_ctx_span[1]])
        if attempt_len > max_len:
            if attempt_ctx_span[1] - attempt_ctx_span[0] == 1:
                # means the dialogue itself has already exceeded the maximum length
                raise ValueError("Dialogue too long")
            break
        else:
            ctx_span = attempt_ctx_span
            covered_roleid2pos.update({roleid: pos})
    
    ctx_span = expand_ctx(ctx_span, tokens, max_len)
    start, end = ctx_span

    # the sentence index of the dialogue within the candidate specific segment
    dialogue_idx = SSWS - start

    return {
        'text': [''.join(token_list) for token_list in tokens[start:end]],
        'dialogue_idx': dialogue_idx,
        'select_sent_idx_span': ctx_span,
        'covered_roleid2pos': covered_roleid2pos
    }


def extract_instances_from_json(split_json_file, max_len):
    """Extract instances from the preprocessed json file,
    and collect raw sentences by the way.
    """
    split_name = split_json_file.split('/')[-1].split('.')[0]
    with open(split_json_file, encoding='utf-8') as fin:
        instances = json.load(fin)
    
    sind_instances = OrderedDict()
    for inst_id, instance in instances.items():
        raw_sents = instance['context_pre'] + [instance['quote']] + instance['context_next']

        speaker_name = instance['entity']
        candidates = instance['candidate']
        if speaker_name not in candidates:
            candidates.append(speaker_name)

        alias2id = {name: str(idx) for idx, name in enumerate(candidates)}
        speaker_id = alias2id[speaker_name] if speaker_name in alias2id else '-1'

        tokens, roleid2poses, roleid2idx = convert_mentions_to_signs(raw_sents, alias2id, max_len)
        feature = build_instance(tokens, roleid2poses, max_len)

        if speaker_id not in feature['covered_roleid2pos']:
            continue

        text = feature['text']
        start, end = feature['select_sent_idx_span']
        target_idx = feature['dialogue_idx']

        # roleid2poses in the selected sentences
        covered_roleid2poses = {}
        for role_id, poses in roleid2poses.items():
            covered_roleid2poses[role_id] = [(pos[0] - start, pos[1]) for pos in poses if pos[0] >= start and pos[0] < end]

        # get neighbour utterances' speaker ids
        speaker_ids = [speaker_id]
        dialogue_idxes = [target_idx]
            
        instance_id = f'{split_name}-{inst_id}'
        sind_instances[instance_id] = {
            'ws': SSWS,
            'text': text,
            'speaker_ids': speaker_ids,
            'dialogue_idxes': dialogue_idxes,
            'target_idx': target_idx,
            'roleid2idx': roleid2idx,
            'roleid2poses': covered_roleid2poses,
        }

    print(f"Number of speaker identification instances obtained from {split_json_file}:", len(sind_instances))

    return sind_instances


def process_split(split_json_path):
    """Obtain samples from a json file
    """
    split_name = split_json_path.split('/')[-1].split('.')[0]
    sind_instances = extract_instances_from_json(split_json_path, args.max_len)
    
    instance_output_path = os.path.join(os.path.dirname(split_json_path), f'{split_name}_instances.json')
    with open(instance_output_path, 'w', encoding='utf-8') as fout:
        json.dump(sind_instances, fout, ensure_ascii=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./jy_data')
    parser.add_argument('--ssws', type=int, default=5, help="Single side window size (SSWS), means the number of sentences on each side of the target utterance")
    parser.add_argument('--dist-in-dlg', type=int, default=100)
    parser.add_argument('--max-len', type=int, default=480)
    args = parser.parse_args()

    SSWS = args.ssws
    DIST_IN_DLG = args.dist_in_dlg
    MAX_LEN = args.max_len

    for split_name in os.listdir(args.data_dir):
        if '.' in split_name:
            continue
        process_split(os.path.join(args.data_dir, f'{split_name}/{split_name}.json'))

    with open(os.path.join(args.data_dir, 'dataset_args.json'), 'w') as fout:
        json.dump(vars(args), fout, indent=4)

