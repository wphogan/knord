import argparse
import json
import os
import string

import torch
from nltk.corpus import stopwords
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForMaskedLM

from clustering.data import RETestDataset

SPECIAL_TOKEN = {'<s>', '</s>', '<pad>', '<mask>', "[object_start]", "[object_end]", "[subject_start]",
                 "[subject_end]"} | set(string.punctuation) | set(stopwords.words('english'))


def parse_args():
    parser = argparse.ArgumentParser(description="OWRelation")
    parser.add_argument(
        "--model_name",
        type=str,
        default='./results',
        help="The name of the model.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Directory of dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='results',
        help="Directory of outputs.",
    )
    parser.add_argument(
        "--unlabel_data",
        type=str,
        default=None,
        help="Unlabeled data.",
    )
    parser.add_argument(
        "--label_data",
        type=str,
        default=None,
        help="Unlabeled data.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda:0',
        help="Device of learning.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Batch size (per device) for the training dataloader.",
    )
    args = parser.parse_args()

    return args


def filter(s):
    s = s.lower()
    if s not in SPECIAL_TOKEN and len(s) > 2:
        return True
    else:
        return False


def target_ranking(predicted_ids, target_ids, tokenizer):
    answers = []
    for predicted, target in zip(predicted_ids, target_ids):
        predicted = predicted.tolist()
        target = set(target.tolist())

        temp = []

        for id in predicted:
            if id in target:
                temp.append(id)

        temp = tokenizer.convert_ids_to_tokens(temp)
        temp = [token.strip('\u0120') for token in temp]
        temp = [token for token in temp if filter(token)]
        answers.append(temp)

    return answers


def _ranking(predicted_ids, tokenizer, topk=10):
    answers = []
    for predicted in predicted_ids:
        predicted = predicted.tolist()[:topk]

        predicted = tokenizer.convert_ids_to_tokens(predicted)
        predicted = [token.strip('\u0120') for token in predicted]
        answers.append(predicted)

    return answers


def main():
    args = parse_args()

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = RobertaForMaskedLM.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    eval_data = RETestDataset(args.unlabel_data, tokenizer, args, is_eval=True)
    eval_dataloader = DataLoader(eval_data, collate_fn=eval_data.collate_fn, batch_size=args.batch_size)

    f = open(os.path.join(args.output_dir, 'test_result_train+test.json'), 'w')

    for batch in tqdm(eval_dataloader):

        features, relation_pos, raw_data = batch

        for k, v in features.items():
            features[k] = v.to(args.device)

        with torch.no_grad():
            outputs = model(**features)
            logits = outputs.logits.detach().cpu()

        mask_logits = []
        for i in range(len(relation_pos)):
            mask_logits.append(logits[i][relation_pos[i]])

        mask_logits = torch.stack(mask_logits)

        predicted_token_ids = torch.argsort(mask_logits, dim=-1, descending=True)

        in_sentence_tokens = target_ranking(predicted_token_ids, features['input_ids'].cpu(), tokenizer)
        all_tokens = _ranking(predicted_token_ids, tokenizer)

        for raw_datum, in_token, all_token in zip(raw_data, in_sentence_tokens, all_tokens):
            f.write(json.dumps({'raw_data': raw_datum, 'in_sent_prediction': in_token, 'prediction': all_token}) + '\n')

    train_data = RETestDataset(args.label_data, tokenizer, args, is_eval=True)
    train_dataloader = DataLoader(train_data, collate_fn=eval_data.collate_fn, batch_size=args.batch_size)

    for batch in tqdm(train_dataloader):

        features, relation_pos, raw_data = batch

        for k, v in features.items():
            features[k] = v.to(args.device)

        with torch.no_grad():
            outputs = model(**features)
            logits = outputs.logits.detach().cpu()

        mask_logits = []
        for i in range(len(relation_pos)):
            mask_logits.append(logits[i][relation_pos[i]])

        mask_logits = torch.stack(mask_logits)

        predicted_token_ids = torch.argsort(mask_logits, dim=-1, descending=True)

        in_sentence_tokens = target_ranking(predicted_token_ids, features['input_ids'].cpu(), tokenizer)
        all_tokens = _ranking(predicted_token_ids, tokenizer)

        for raw_datum, in_token, all_token in zip(raw_data, in_sentence_tokens, all_tokens):
            f.write(json.dumps({'raw_data': raw_datum, 'in_sent_prediction': in_token, 'prediction': all_token}) + '\n')


if __name__ == "__main__":
    main()
