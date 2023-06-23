import os
from collections import Counter
from tqdm import tqdm
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.utils_gpt import GPTExamples
from utils.utils_gpt import load_jsonl_data


def p_r_f1_f1cls(preds, targets):
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()

    for i in range(len(preds)):
        guess = preds[i]
        gold = targets[i]
        if gold == 0 and guess == 0:
            continue
        if gold == 0 and guess != 0:
            guess_by_relation[guess] += 1
        if gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        if gold != 0 and guess != 0:
            guess_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[gold] += 1
    prec, recall, micro_f1 = 0, 0, 0

    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())
        micro_f1 = 2 * recall * prec / (recall + prec)
    return prec, recall, micro_f1


def get_cls_representation(outputs):
    lhs = outputs.hidden_states[-1]
    return lhs[:, 0, :]


def main():
    # Load data
    dataset_name = 'fewrel'
    gpt_predictions = load_jsonl_data(f'data/{dataset_name}/{dataset_name}_unlabeled_gpt_predictions.jsonl')

    # Load helpers
    gpt_examples = GPTExamples(dataset=dataset_name)
    all_rel_names_natural = gpt_examples.all_rel_names
    all_rel_names_gt = gpt_examples.rel2id.keys()
    if dataset_name == 'fewrel':
        all_rel_names_gt = gpt_examples.all_rel_names

    # Load model and cos
    model_name = 'microsoft/deberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    # Tokenize / embed all relation names
    tokenized_gt_rel_classes = tokenizer(all_rel_names_natural, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs_classes = model(**tokenized_gt_rel_classes, output_hidden_states=True)
        classes_cls = get_cls_representation(outputs_classes)

    # Tokenize / embed / + map all GPT predictions
    mapped_predictions = []
    ground_truth = []
    is_known = []
    for entry in tqdm(gpt_predictions):
        prediction = entry['response']
        ground_truth.append(entry['rel_name'])
        is_known.append(entry['is_known'])

        # 1. Is prediction in GT classes? If yes, do not map
        if prediction in all_rel_names_gt:
            mapped_predictions.append(prediction)

        # 2. If not, map to GT class
        else:
            with torch.no_grad():
                tokenized_pred = tokenizer(prediction, padding=True, truncation=True, return_tensors="pt")
                outputs_pred = model(**tokenized_pred, output_hidden_states=True)
            pred_cls = get_cls_representation(outputs_pred).expand(classes_cls.shape[0], -1)

            # Compute cosine similarity
            sim = cos(classes_cls, pred_cls)

            # Get top 1
            top1 = torch.topk(sim, 1)
            top1_gt_name = [gpt_examples.id2rel[i.item()] for i in top1.indices]
            top1_natural_lang_name = [all_rel_names_natural[i] for i in top1.indices]
            top1_score = top1.values
            mapped_predictions.append(top1_gt_name[0])

    # Compute metrics
    all_pred, known_pred, novel_pred = [], [], []
    all_gt, known_gt, novel_gt = [], [], []
    for pred, gt, is_known in zip(mapped_predictions, ground_truth, is_known):
        all_pred.append(pred)
        all_gt.append(gt)
        if is_known:
            known_pred.append(pred)
            known_gt.append(gt)
        else:
            novel_pred.append(pred)
            novel_gt.append(gt)

    _, _, f1_all = p_r_f1_f1cls(all_pred, all_gt)
    _, _, f1_known = p_r_f1_f1cls(known_pred, known_gt)
    _, _, f1_novel = p_r_f1_f1cls(novel_pred, novel_gt)

    print(f'{f1_all:.03f}\t{f1_known:0.03f}\t{f1_novel:0.03f}')
    print(f'{f1_all:.03f},&,{f1_known:0.03f},&,{f1_novel:0.03f}')

if __name__ == '__main__':
    print('Starting file: ', os.path.basename(__file__))
    main()
    print('\nCompelted file: ', os.path.basename(__file__))
