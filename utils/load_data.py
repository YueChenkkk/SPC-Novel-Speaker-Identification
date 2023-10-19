
import os
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from .get_data import get_samples


class SindDataset(Dataset):
    def __init__(self, data_dict, sort):
        super().__init__()
        self.data_dict = data_dict
        if sort:
            self.sample_indexes = self.get_desc_sorted_indexes()
        else:
            self.sample_indexes = list(range(self.__len__()))
        
    def __getitem__(self, index):
        return {k: v[self.sample_indexes[index]] for k, v in self.data_dict.items()}

    def __len__(self):
        return len(self.data_dict['instance_id'])

    def get_desc_sorted_indexes(self):
        def text_len(index):
            return len(self.data_dict['input_ids'][index])

        sorted_indexes = sorted(range(self.__len__()), key=lambda x: -text_len(x))
        return sorted_indexes


def get_dataloader(split_name, tokenizer, args):
    """Generate the dataloader for this epoch
    """
    if split_name == 'train':
        skip_only_one_candidate = True
    else:
        skip_only_one_candidate = False
        role_mask_prob = 0.0

    samples = get_samples(
        split_name, 
        tokenizer=tokenizer, 
        role_mask_prob=args.role_mask_prob, 
        skip_only_one_candidate=skip_only_one_candidate, 
        args=args
    )
    dataset = SindDataset(samples, sort=True)
    sampler = DistributedSampler(
        dataset,
        shuffle=False, 
        drop_last=True
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size_per_gpu,
        sampler=sampler,
        collate_fn=lambda x: data_collator(x, tokenizer=tokenizer, max_len=args.max_len)
    )

    return data_loader


def _pad_inputs_and_mask(input_ids, pad_id, max_len):
    batch_size = len(input_ids)
    input_lens = [len(x) for x in input_ids]
    pad_len = min(max(input_lens), max_len)

    pad_input_ids = torch.LongTensor(batch_size, pad_len).fill_(pad_id)
    for i, x in enumerate(input_ids):
        copy_len = min(len(x), max_len)
        pad_input_ids[i][:copy_len].copy_(torch.tensor(x[:copy_len]).long())
    token_type_ids = torch.zeros_like(pad_input_ids)
    attention_mask = [[1] * min(l, max_len) + [0] * (pad_len - l) for l in input_lens]
    attention_mask = torch.LongTensor(attention_mask)

    bert_inputs = {
        'input_ids': pad_input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
    }
    return bert_inputs, input_lens


def data_collator(samples, tokenizer, max_len):
    instance_ids = [sample['instance_id'] for sample in samples]
    input_ids = [sample['input_ids'] for sample in samples]
    idx2roleid = [sample['idx2roleid'] for sample in samples]
    target_token_ids = [sample['target_token_ids'] for sample in samples]
    distance = [sample['distance'] for sample in samples]

    mask_pos = [sample['mask_pos'] for sample in samples]
    label_idx = [sample['label_idx'] for sample in samples]
    neighbour_utter_mask_poses = [sample['neighbour_utter_mask_poses'] for sample in samples]
    neighbour_utter_label_idxes = [sample['neighbour_utter_label_idxes'] for sample in samples]
    narr_role_mask_poses = [sample['role_mask_poses'] for sample in samples]
    narr_role_label_idxes = [sample['role_label_idxes'] for sample in samples]

    pad_id = tokenizer.convert_tokens_to_ids('[PAD]')
    bert_inputs, input_lens = _pad_inputs_and_mask(input_ids, pad_id, max_len)    

    forward_inputs = {
        'bert_inputs': bert_inputs,
		'target_token_ids': target_token_ids,
		'mask_pos': mask_pos,
        'label_idx': label_idx,
        'neighbour_utter_mask_poses': neighbour_utter_mask_poses,
        'neighbour_utter_label_idxes': neighbour_utter_label_idxes,
        'narr_role_mask_poses': narr_role_mask_poses,
        'narr_role_label_idxes': narr_role_label_idxes,
    }

    other_info = {
        'instance_id': instance_ids,
        'idx2roleid': idx2roleid,
        'distance': distance
    }
    return forward_inputs, other_info


