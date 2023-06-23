from collections import Counter, defaultdict

import torch.nn.functional as F
from torch.nn import KLDivLoss


def kl_div_loss_custom(v_s, v_d):
    kl_loss = KLDivLoss(reduction="batchmean", log_target=True)
    v_s_softmax = F.log_softmax(v_s, dim=-1)
    v_d_softmax = F.log_softmax(v_d, dim=-1)
    return kl_loss(v_s_softmax, v_d_softmax)


def p_r_f1_f1cls(preds, targets, rel_ids):
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()

    no_rel_stats = defaultdict(int)

    for i in range(len(preds)):
        guess = preds[i]
        gold = targets[i]
        if gold == 0 and guess == 0:
            # NOREL: True positive
            no_rel_stats['TP'] += 1
            continue
        if gold == 0 and guess != 0:
            # NOREL: False negative
            no_rel_stats['FN'] += 1
            guess_by_relation[guess] += 1
        if gold != 0 and guess == 0:
            # NOREL: False positive
            no_rel_stats['FP'] += 1
            gold_by_relation[gold] += 1
        if gold != 0 and guess != 0:
            guess_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[gold] += 1


    f1_by_relation = Counter()
    recall_by_relation = Counter()
    prec_by_relation = Counter()

    no_rel_prec = (no_rel_stats['TP'] + no_rel_stats['FP']) and no_rel_stats['TP'] / (no_rel_stats['TP'] + no_rel_stats['FP']) or 0
    no_rel_recall = (no_rel_stats['TP'] + no_rel_stats['FN']) and no_rel_stats['TP'] / (no_rel_stats['TP'] + no_rel_stats['FN']) or 0
    no_rel_f1 = (no_rel_prec + no_rel_recall) and 2 * no_rel_prec * no_rel_recall / (no_rel_prec + no_rel_recall) or 0

    if no_rel_stats['TP'] + no_rel_stats['FP'] + no_rel_stats['FN'] > 0 and 1 in rel_ids:
        print('*' * 50)
        print(f'NOREL: P: {no_rel_prec:.3f}, R: {no_rel_recall:.3f}, F1: {no_rel_f1:.3f}')
        print('*' * 50)

    for i in rel_ids:
        if i == 0:
            continue # skip negative class
        recall = 0
        if gold_by_relation[i] > 0:
            recall = correct_by_relation[i] / gold_by_relation[i]
        precision = 0
        if guess_by_relation[i] > 0:
            precision = correct_by_relation[i] / guess_by_relation[i]
        if recall + precision > 0:
            f1_by_relation[i] = 2 * recall * precision / (recall + precision)
        recall_by_relation[i] = recall
        prec_by_relation[i] = precision

    prec, recall, micro_f1 = 0, 0, 0

    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())
        micro_f1 = 2 * recall * prec / (recall + prec)
    return prec, recall, micro_f1, f1_by_relation
