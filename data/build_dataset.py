

from lib2to3.pgen2 import token
import os
import json
import copy
import re
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from argparse import ArgumentParser


stop_names = ['男', '女', '男人', '女人', '男生', '女生', '男孩', '女孩']

parser = ArgumentParser()
parser.add_argument('--data-dir', type=str, default='/xdl/private/chenyue.chen/data/sind_512_label_all_utterances/data_batch1')
parser.add_argument('--ssws', type=int, default=10)
parser.add_argument('--dist-in-dlg', type=int, default=100)
parser.add_argument('--max-len', type=int, default=480)
args = parser.parse_args()

SSWS = args.ssws
DIST_IN_DLG = args.dist_in_dlg
MAX_LEN = args.max_len


def load_char_list(char_list_path):
	with open(char_list_path, 'r', encoding='utf-8') as fin:
		char_list = json.load(fin)
	name_list = []
	alias2id = {}
	id2aliases = {}
	for char in char_list:
		aliases = list(map(lambda x: x[1:-1], char['name'][1:-1].split(',')))
		# aliases = char['name']
		name_list.extend(aliases)
		for alias in aliases:
			alias2id[alias] = (char['id'], char['sex'])
		id2aliases[char['id']] = aliases
	return name_list, alias2id, id2aliases


class SlideWindow:
    """This class generates a sliding window on the sentences in the novel
    """
    def __init__(self, fragments, single_sided_ws):
        self.fragments = fragments
        self._anchor_idx = single_sided_ws - 1
        self._single_sided_ws = single_sided_ws  # single-sided window size

    def get_next_window(self):
        self._anchor_idx += 1
        if self._anchor_idx < self._single_sided_ws:
            leftmost = 0
        else:
            leftmost = self._anchor_idx - self._single_sided_ws
        
        rightmost = self._anchor_idx + self._single_sided_ws
        if rightmost >= len(self.fragments):
            return None, -1

        fragments_in_window = copy.deepcopy(self.fragments[leftmost:rightmost + 1])

        return fragments_in_window, self._anchor_idx


def character_filter(characters_json_file):
    """
    Filter common character aliases, ”男“”女“ for example, as well as aliases 
    which contains digits as indexes.

    params
        characters_json_file: character aliases list file
    """
    name_list, alias2idgdr, id2aliases = load_char_list(characters_json_file)

    filtered_name_list = []
    filtered_alias2idgdr = {}
    for name in name_list:
        if name in stop_names:
            continue
        if re.search(r'[0-9]', name):
            continue
        
        filtered_name_list.append(name)
        filtered_alias2idgdr[name] = alias2idgdr[name]
    
    filtered_id2aliases = {}
    for role_id, aliases in id2aliases.items():
        filtered_id2aliases[role_id] = list(filter(lambda x:x in filtered_name_list, aliases))

    return filtered_name_list, filtered_alias2idgdr, filtered_id2aliases


def judge_slide_window(fragments_in_win, id2aliases):
    '''
    Judge whether the fragments in the window contains the mention of the speaker
    of the dialogue fragment (anchor).

    Params
        fragments_in_win: text fragment covered by the window
        id2aliases: the dict mapping role ID to its aliases (a list)
    '''
    dialogue_fragment = fragments_in_win[SSWS]
    role_id = dialogue_fragment['role']['id']
    if role_id not in id2aliases:
        return False

    alias_list = id2aliases[role_id]

    found_mention = False
    for fragment in fragments_in_win:    
        for segment in fragment['segments']:
            if not segment.startswith('<Mention>'):
                continue
            mention = segment.strip('<Mention>')
            if mention in alias_list:
                found_mention = True
                break
        if found_mention:
            break

    return found_mention


def convert_mentions_to_signs(fragments_in_win, alias2idgdr, max_len):
    '''
    Converts role names to masked names `C[X]`,
    returns the sentences after this conversion and the 
    mappings from role IDs to `C[X]`.
    '''
    tokens = []
    roleid2poses = {}
    roleid2sign = {}

    for f_idx, fragment in enumerate(fragments_in_win):
        new_token_list = []

        for segment in fragment['segments']:

            if not segment.startswith('<Mention>'):
                new_token_list += list(segment)
                continue

            mention = segment.strip('<Mention>')
            if mention not in alias2idgdr:
                new_token_list += list(mention)
                continue

            roleid = alias2idgdr[mention][0]
            if roleid not in roleid2sign:
                roleid2sign[roleid] = f'[C{len(roleid2sign)}]'
                roleid2poses[roleid] = []

            roleid2poses[roleid].append((f_idx, len(new_token_list)))
            new_token_list.append(roleid2sign[roleid])
            
        if len(new_token_list) >= max_len:
            new_token_list = new_token_list[:max_len]

        tokens.append(new_token_list)

    return tokens, roleid2poses, roleid2sign


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


def extract_instances_and_text_from_json(chapter_json_file, characters_json_file):
    """Extract instances from the preprocessed json file,
    and collect raw sentences by the way.
    """
    book_id, chapter_id = chapter_json_file.split('/')[-3:-1]
    with open(chapter_json_file, encoding='utf-8') as fin:
        fragments = json.load(fin)

    _, alias2id, id2aliases = character_filter(characters_json_file)
    
    slwin = SlideWindow(fragments, single_sided_ws=SSWS)

    sind_instances = {}
    while True:
        fragments_in_win, anchor_idx = slwin.get_next_window()
        if not fragments_in_win:
            break

        if not fragments_in_win[SSWS]['role']:
            continue

        speaker_id = fragments_in_win[SSWS]['role']['id']
        dialogue_id = fragments_in_win[SSWS]['id']

        if judge_slide_window(fragments_in_win, id2aliases):
            tokens, roleid2poses, roleid2sign = convert_mentions_to_signs(fragments_in_win, alias2id, MAX_LEN)
            feature = build_instance(tokens, roleid2poses, MAX_LEN)

            if not feature:
                continue

            if len(roleid2sign) < 2:
                continue

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
            speaker_ids = []
            dialogue_idxes = []
            for idx, fragment in enumerate(fragments_in_win[start:end]):
                if not fragment['role']:
                    continue

                speaker_id = fragment['role']['id']
                if speaker_id not in roleid2sign:
                    continue

                speaker_ids.append(speaker_id)
                dialogue_idxes.append(idx)

            assert target_idx in dialogue_idxes, "Target dialogue not involved"
                
            instance_id = f'{book_id}-{chapter_id}-{dialogue_id}'
            sind_instances[instance_id] = {
                'anchor_idx': anchor_idx,
                'ws': SSWS,
                'text': text,
                'speaker_ids': speaker_ids,
                'dialogue_idxes': dialogue_idxes,
                'target_idx': target_idx,
                'roleid2sign': roleid2sign,
                'roleid2poses': covered_roleid2poses,
            }

    return sind_instances


def run_single_process(book_dir, data_dir):
    """Run a single process to obtain samples from a book
    """
    chapter_dirs = []
    characters_json_path = ''
    for item_name in os.listdir(os.path.join(data_dir, book_dir)):
        if os.path.splitext(item_name)[1] == '':
            # is chapter directory
            chapter_json_path = os.path.join(data_dir, book_dir, item_name, item_name + '_preproc.json')
            if os.path.exists(chapter_json_path):
                chapter_dirs.append(chapter_json_path)
        elif item_name == f'{book_dir}.json':
            characters_json_path = os.path.join(data_dir, book_dir, item_name)
    
    assert characters_json_path

    for chapter_json_path in chapter_dirs:
        sind_instances = extract_instances_and_text_from_json(
            chapter_json_path, 
            characters_json_path
        )
        
        instance_output_path = os.path.join(
            os.path.dirname(chapter_json_path), 
            chapter_json_path.split('/')[-2] + '_instances.json'
        )

        with open(instance_output_path, 'w', encoding='utf-8') as fout:
            json.dump(sind_instances, fout, ensure_ascii=False)


def test_example(test_instance_file):
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('/xdl/private/chenyue.chen/pretrained/chinese-roberta-wwm-ext-large')
    tokenizer.add_special_tokens({'additional_special_tokens': [f'[C{i}]' for i in range(50)]})

    with open(test_instance_file, encoding='utf-8') as fin:
        instance_dict = json.load(fin)

    instance_ids = list(instance_dict.keys())
    for idx, i in enumerate([10, 100]):
        print(f'#############\n# Example {idx + 1} #\n#############')

        test_instance = instance_dict[instance_ids[i]]
        anchor_idx = test_instance['anchor_idx']
        target_idx = test_instance['target_idx']
        dialogue_idxes = test_instance['dialogue_idxes']

        roleid2sign = test_instance['roleid2sign']
        speakers = [roleid2sign.get(x, '[C0]') for x in test_instance['speaker_ids']]

        print(f"Role ID to sign: \n{roleid2sign}")
        print(f"Labelled sentence indexes: \n{dialogue_idxes}")
        print(f"Target index: \n{target_idx}")
        print(f"Speakers: \n{speakers}")

        sents = test_instance['text']
        print("Instance text:")
        for line in enumerate(sents):
            print(line)



dirs_under_data_dir = list(filter(lambda x: '.' not in x, os.listdir(args.data_dir)))

print("Multi-processes started...")
with mp.Pool(processes=8) as p:
    result = list(
        tqdm(
            p.imap(
                partial(
                    run_single_process, 
                    data_dir=args.data_dir
                ), 
                dirs_under_data_dir
            ),
            total=len(dirs_under_data_dir)
        )
    )
print("Multi-processes finished >_<")

# for book_id in dirs_under_data_dir:
#     run_single_process(book_id, args.data_dir)

with open(
    os.path.join(os.path.dirname(args.data_dir), 'dataset_args.json'),
    'w',
    encoding='utf-8'
) as fout:
    json.dump(vars(args), fout, indent=4)


# # Test
test_book_dir = '17309'
test_chapter_dir = '51759'
test_dir = os.path.join(args.data_dir, test_book_dir, test_chapter_dir)
test_example(os.path.join(test_dir, test_chapter_dir + '_instances.json'))

# counter = 0
# for dirname in dirs_under_data_dir:
#     book_dir = os.path.join(data_dir, dirname)
#     for chapter_dir in list(filter(lambda x: '.' not in x, os.listdir(book_dir))):
#         if not os.path.exists(os.path.join(book_dir, chapter_dir, chapter_dir + '_instances.json')):
#             counter += 1

# print(counter)


"""
{
    "worker": {
        "cpuSize": 64,
        "memory": 50,
        "gpuSize": 0
    },
    "imageUrl": "harbor.ximalaya.local/xdl/chenyue.chen:202201201821",
    "jobPath": "/xdl/private/chenyue.chen/codes/sind/src/develop/speaker_identification/psi/datasets/build_dataset.py",
    "parameter": {}
}
"""

