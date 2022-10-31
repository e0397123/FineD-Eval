"""Adapted evaluation script from the MRQA Workshop Shared Task.
Adapted from the SQuAD v1.1 official evaluation script.
"""
import argparse
import collections
from collections import Counter
import json
import re
import string
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.stats import spearmanr


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def score_predictions_dialogue_flatten(all_predictions):
    label_list = [p['label'] for p in all_predictions]
    print(label_list[:20])
    preds = [p['score'] for p in all_predictions]
    print(preds[:20])
    acc = accuracy_score(label_list, preds)

    predictions = []
    for p in all_predictions:
        predictions.append(p['pred'])
    report = {"acc":acc, "loss": np.mean([p['loss'] for p in all_predictions])}
    expert_predictions = []
    expert_loss = []
    report["experts"] = {}
    if "experts_name" in all_predictions[0]:
        for e in all_predictions[0]["experts_name"]:
            report["experts"][e] = {'loss': []}
    for p in all_predictions:
        if "experts_pred" in p:
            expert_predictions.append(p["experts_pred"])
        if "experts_loss" in p:
            expert_loss.append(p["experts_loss"])
            for x, y in zip(p["experts_name"], p["experts_loss"]):
                report["experts"][x]['loss'].append(y)
    report["experts"] = {k: {'loss': np.mean(v['loss'])} for k, v in report['experts'].items()}

    return report, predictions, expert_predictions, expert_loss


def average_dicts(dicts, short=False):
    avg = {}
    for k, v in dicts[0].items():
        vs = [d[k] for d in dicts if k in d]
        if type(v) == dict:
            avg[k] = average_dicts(vs)
        elif type(v) in (int, float, np.float64):
            avg[k] = np.mean(vs)
            avg[f"{k}_std"] = np.std(vs)
            if not short:
                avg[f"{k}_lst"] = vs
        elif type(vs[0]) == list and len(vs[0]) == 1:
            avg[k] = [v[0] for v in vs]
        else:
            avg[k] = vs
    return avg
