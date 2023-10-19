
import os
import json
import copy
import jieba
import torch
from tqdm import tqdm
from argparse import ArgumentParser


def nearest_mention_location(raw_sents, mention_positions):
    """
    Nearest Mention Location
    
    return
        The position of the mention which is the nearest to the quote.
    """
    ws = SSWS

    def word_dist(pos):
        """
        The word level distance between quote and the mention position

        param
            pos: [sentence-level index, word-level index] of the character mention.

        return
            w_d: word-level distance between the mention and the quote.
        """
        if pos[0] == ws:
            w_d = ws * 2
        elif pos[0] < ws:
            w_d = sum(len(sent) for sent in raw_sents[pos[0] + 1:ws]) + len(raw_sents[pos[0]]) - pos[1]
        else:
            w_d = sum(len(sent) for sent in raw_sents[ws + 1:pos[0]]) + pos[1]
        return w_d
    
    sorted_positions = sorted(mention_positions, key=lambda x: word_dist(x))
    nearest_pos = sorted_positions[0]

    # a little trick
    if raw_sents[ws - 1][-1] == '：':
        # if the preceding sentence ends with '：'
        for pos in sorted_positions:
            # search candidate mention from left-side context
            if pos[0] < ws:
                nearest_pos = pos
                break

    return nearest_pos, word_dist(nearest_pos)


def convert_mentions_to_masks(raw_sents_in_list, max_len):
    '''Converts role names to masked names `C[X]`
    '''
    tokens = []
    roleid2poses = {}
    roleid2idx = {}

    for sent_idx, sent in enumerate(raw_sents_in_list):
        new_token_list = []
        seg_sent = list(jieba.cut(sent, cut_all=False))

        for word_idx, word in enumerate(seg_sent):
            if word in alias2id:
                pos_in_sent = len(new_token_list)
                role_id = alias2id[word]

                if role_id not in roleid2idx:
                    roleid2idx[role_id] = f'[C{len(roleid2idx)}]'
                    roleid2poses[role_id] = []

                roleid2poses[role_id].append((sent_idx, pos_in_sent))
                new_token_list.append(roleid2idx[role_id])
            else:
                new_token_list += list(word)
            
        if len(new_token_list) > max_len:
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


def load_data_from_file(data_file):
    """
    Build the dataloader for training.

    Input:
        data_file: labelled training data as in https://github.com/YueChenkkk/Chinese-Dataset-Speaker-Identification.
        name_list_path: the path of the name list which contains the aliases of characters.
        args: parsed arguments.
        skip_only_one: a flag for filtering out the instances that have only one candidate, such 
            instances have no effect while training.

    Output:
        data_list: [(sentences, speaker name, instance category)]
    """
    # load instances from file
    with open(data_file, 'r', encoding='utf-8') as fin:
        data_lines = fin.readlines()

    # pre-processing
    data_list = []

    for i, line in enumerate(tqdm(data_lines, total=len(data_lines))):
        offset = i % 26
        
        if offset == 0:
            raw_sents_in_list = []
            continue

        if offset < 22:
            raw_sents_in_list.append(line.strip())

        if offset == 22:
            speaker_name = line.strip().split()[-1]

        if offset == 24:
            category = line.strip().split()[-1]
            data_list.append((raw_sents_in_list, speaker_name, category))

    return data_list


def match_sents(left_sents, right_sents):
    """Match two groups of sentences, return their distance 
    if they have sentences in common.
    """
    rightmost = int(SSWS * 2 + 1)

    for idx1, sent1 in enumerate(left_sents):
        for idx2, sent2 in enumerate(right_sents):
            if sent1 != sent2:
                continue

            dist = idx1 - idx2
            bad = False
            for i in range(idx1, rightmost + min(0, dist)):
                if left_sents[i] != right_sents[i - dist]:
                    bad = True
                    break

            if bad:
                continue

            return dist
    
    return None


def retrieve_neighbour_utterances(data_list, inst_idx, target_idx, ctx_span, roleid2idx):
    """Recall neighborhood utterances for data_list[inst_idx]
    """
    dialogue_idxes = []
    speaker_ids = []

    cur_sents, cur_spk_name, _ = data_list[inst_idx]

    left = max(inst_idx - SSWS, 0)
    right = min(len(data_list), inst_idx + SSWS + 1)

    for iter_sents, iter_spk_name, _ in data_list[left:right]:
        dist = match_sents(iter_sents, cur_sents)

        if dist == None:
            continue

        spk_id = alias2id[iter_spk_name]
        dlg_idx = SSWS - dist

        if spk_id not in roleid2idx or dlg_idx < ctx_span[0] or dlg_idx > ctx_span[1] - 1:
            continue

        dialogue_idxes.append(target_idx - dist)
        speaker_ids.append(spk_id)

    return dialogue_idxes, speaker_ids


def build_data_dict(data_list, max_len, stage, filter_only_one_candidate=False):
    sind_instances = {}

    for inst_idx, (raw_sents, speaker_name, category) in enumerate(data_list):
        instance_id = f'{stage}-{inst_idx}'

        speaker_id = alias2id[speaker_name]

        tokens, roleid2poses, roleid2idx = convert_mentions_to_masks(raw_sents, max_len)
        feature = build_instance(tokens, roleid2poses, max_len)
        ctx_span = feature['select_sent_idx_span']
        target_idx = feature['dialogue_idx']

        if not feature:
            continue

        if speaker_id not in feature['covered_roleid2pos']:
            continue

        if filter_only_one_candidate and len(feature['covered_roleid2pos']) == 1:
            continue

        dialogue_idxes, speaker_ids = retrieve_neighbour_utterances(data_list, inst_idx, target_idx, ctx_span, roleid2idx)

        covered_roleid2poses = {}
        for role_id, poses in roleid2poses.items():
            covered_roleid2poses[role_id] = [(pos[0] - ctx_span[0], pos[1]) for pos in poses if pos[0] >= ctx_span[0] and pos[0] < ctx_span[1]]

        sind_instances[instance_id] = {
            'ws': SSWS,
            'text': feature['text'],
            'speaker_ids': speaker_ids,
            'dialogue_idxes': dialogue_idxes,
            'target_idx': target_idx,
            'roleid2idx': roleid2idx,
            'roleid2poses': covered_roleid2poses
        }

    return sind_instances


def merge_dict(sum_dict, to_be_merged):
    for key in to_be_merged:
        if key not in sum_dict:
            sum_dict[key] = to_be_merged[key]
        else:
            sum_dict[key].update(to_be_merged[key])


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./wp_data')
    parser.add_argument('--ssws', type=int, default=10, help="Single side window size (SSWS), means the number of sentences on each side of the target utterance")
    parser.add_argument('--dist-in-dlg', type=int, default=100)
    parser.add_argument('--max-len', type=int, default=480)
    args = parser.parse_args()

    SSWS = args.ssws
    DIST_IN_DLG = args.dist_in_dlg
    MAX_LEN = args.max_len

    # load character list from file
    with open(os.path.join(args.data_dir, 'name_list.txt'), 'r') as fin:
        name_lines = fin.readlines()

    alias2id = {}
    id2alias = []
    for i, line in enumerate(name_lines):
        for alias in line.strip().split()[1:]:
            alias2id[alias] = str(i)
            jieba.add_word(alias)
        id2alias.append(line.strip().split()[1])

    stages = ['train', 'dev', 'test']
    for stage in stages:
        data_txt_file = os.path.join(args.data_dir, f'{stage}/{stage}_unsplit.txt')

        data_list = load_data_from_file(data_txt_file)
        data_dict = build_data_dict(data_list, MAX_LEN, stage, filter_only_one_candidate=(stage=='train'))
        print(f"Number of speaker identification instances obtained from {data_txt_file}:", len(data_dict))

        with open(os.path.join(args.data_dir, f'{stage}/{stage}_instances.json'), 'w') as fout:
            json.dump(data_dict, fout, ensure_ascii=False)

    with open(os.path.join(args.data_dir, 'dataset_args.json'), 'w') as fout:
        json.dump(vars(args), fout, indent=4)