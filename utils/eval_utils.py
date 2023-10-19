
import os
import json
from collections import OrderedDict

from .file_utils import load_json


def update_pred(pred_dict, logits, instance_ids, role_ids):
    """Log the predictions in a batch.
    """
    for score, inst_id, role_id in zip(logits, instance_ids, role_ids):
        new_score = {role_id: score.item()}
        if inst_id in pred_dict:
            pred_dict[inst_id].update(new_score)
        else:
            pred_dict[inst_id] = new_score


def summarize_preds(gathered_preds):
    """Summarize the predictions from different processes.
    """
    summarized_preds = gathered_preds[0]

    for pred_dict in gathered_preds[1:]:
        for inst_id in pred_dict:
            if inst_id not in summarized_preds:
                summarized_preds[inst_id] = {}
            summarized_preds[inst_id].update(pred_dict[inst_id])

    return summarized_preds


def compute_acc(pred_dict, target_dict, bookid2instanceids=None):
    """Compute the testing accuracy.
    """
    n_correct = n_total = 0
    for instance_id, res in pred_dict.items():
        if instance_id not in target_dict:
            continue

        gold_role_id = target_dict[instance_id]
        pred_role_id = max(res.items(), key=lambda x:x[1])[0]

        correct = 1 if pred_role_id == gold_role_id else 0
        n_correct += correct
        n_total += 1

    acc = n_correct / n_total
    return acc, n_total

