

import re
import os
import json
import torch
from transformers import PreTrainedModel

CONFIG_FILE = 'config.json'
MODEL_FILE = 'pytorch_model.bin'
STATE_FILE = 'state.bin'
METRICS_FILE = 'metrics.json'
ARG_FILE = 'args.json'
LOG_FILE = 'training_log.log'


def save_checkpoint(
    output_dir,
    config,
    tokenizer,
    model, 
    optimizer,
    lr_scheduler,
    epoch,
    patience,
    metrics,
    args
):
    # save config
    config.save_pretrained(output_dir)

    # save tokenizer
    tokenizer.save_pretrained(output_dir)

    # save model
    if isinstance(model, PreTrainedModel):
        model.save_pretrained(output_dir)
    else:
        model.module.save_pretrained(output_dir)

    # save training state
    train_state = {
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'best_valid_acc': metrics['best_valid_acc'],
        'patience_counter': patience
    }
    torch.save(train_state, os.path.join(output_dir, STATE_FILE))
    
    # save arguments
    with open(os.path.join(output_dir, ARG_FILE), 'w', encoding='utf-8') as fout:
        json.dump(args.__dict__, fout, indent=4)

    # save metrics
    with open(os.path.join(output_dir, METRICS_FILE), 'w', encoding='utf-8') as fout:
        json.dump(metrics, fout, indent=4)


def filter_params(state_dict, num_hidden_layer):
    filtered_state_dict = {}
    for n, p in state_dict.items():
        # remove redundant layers
        match = re.search(r'[0-9]+', n)
        if match:
            layer_index = int(match.group(0))
            if layer_index >= num_hidden_layer:
                continue

        # the pooler layer is also removed
        if re.search(r'(pooler)|(cls)', n):
            continue

        filtered_state_dict[n] = p

    return filtered_state_dict


def load_train_state(checkpoint_dir, device):
    try:
        train_state = torch.load(checkpoint_dir, map_location=device)
    except OSError:
        train_state = {}

    return train_state


def resume_training_state(
    train_state,
    optimizer,
    lr_scheduler
):
    optimizer.load_state_dict(train_state['optimizer'])
    lr_scheduler.load_state_dict(train_state['lr_scheduler'])


def get_env_params():
    log_json = {}
    for env_name in os.environ.keys():
        if env_name.startswith('JOB_PARAMS_'):
            log_json[env_name.strip('JOB_PARAMS_')] = os.environ.get(env_name)
    return log_json
