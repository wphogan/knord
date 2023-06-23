import argparse
import json
from collections import defaultdict, Counter

import torch
from tqdm import tqdm
from uctopic import UCTopic, UCTopicTokenizer

from clustering.utils import get_GMM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--pca", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=120)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--p", type=float, default=0.3)
    args = parser.parse_args()
    return args


ARGS = parse_args()
DEVICE = torch.device('cpu' if ARGS.gpu is None else f'cuda:{ARGS.gpu}')


def get_uctopic_features(data, label_dict):
    tokenizer = UCTopicTokenizer.from_pretrained("uctopic-base")
    model = UCTopic.from_pretrained("uctopic-base")
    model.to(DEVICE)
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():

        for i in tqdm(range(0, len(data), ARGS.batch_size), ncols=100, desc='Generate all features...'):

            batch = data[i:i + ARGS.batch_size]

            batch_features = []

            for k in range(ARGS.topk):

                text_batch = []
                span_batch = []

                for data_line in batch:
                    text = ' '.join(data_line['tokens'])
                    if k >= len(data_line['in_sent_words']):
                        k = -1
                    if len(data_line['in_sent_words']) > 0:
                        span_text = data_line['in_sent_words'][k]
                        span_start = text.find(span_text)
                        span_end = span_start + len(span_text)
                    else:
                        span_start = 0
                        span_end = 1

                    text_batch.append(text)
                    span_batch.append([(span_start, span_end)])

                inputs = tokenizer(text_batch, entity_spans=span_batch, padding=True, add_prefix_space=True,
                                   return_tensors="pt")

                for k, v in inputs.items():
                    inputs[k] = v.to(DEVICE)

                luke_outputs, entity_pooling = model(**inputs)
                entity_pooling = entity_pooling.squeeze().detach().cpu()
                batch_features.append(entity_pooling)

            batch_features = torch.mean(torch.stack(batch_features, dim=1), dim=1)
            all_features.append(batch_features)

            label_batch = [label_dict[data_line['relation']] for data_line in batch]
            all_labels += label_batch

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.LongTensor(all_labels)

    return all_features, all_labels


def get_uctopic_features_all_pred(data, label_dict):
    tokenizer = UCTopicTokenizer.from_pretrained("uctopic-base")
    model = UCTopic.from_pretrained("uctopic-base")
    model.to(DEVICE)
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():

        for i in tqdm(range(0, len(data), ARGS.batch_size), ncols=100, desc='Generate all features...'):

            batch = data[i:i + ARGS.batch_size]

            batch_features = []

            for k in range(ARGS.topk):

                text_batch = []
                span_batch = []

                for data_line in batch:
                    if k >= len(data_line['all_pred']):
                        k = -1

                    text = data_line['all_pred'][k]

                    if len(data_line['all_pred']) > 0:
                        span_text = data_line['all_pred'][k]
                        span_start = 0
                        span_end = len(span_text)
                    else:
                        span_start = 0
                        span_end = 1

                    text_batch.append(text)
                    span_batch.append([(span_start, span_end)])

                inputs = tokenizer(text_batch, entity_spans=span_batch, padding=True, add_prefix_space=True,
                                   return_tensors="pt")

                for k, v in inputs.items():
                    inputs[k] = v.to(DEVICE)

                luke_outputs, entity_pooling = model(**inputs)
                entity_pooling = entity_pooling.squeeze().detach().cpu()
                batch_features.append(entity_pooling)

            batch_features = torch.mean(torch.stack(batch_features, dim=1), dim=1)
            all_features.append(batch_features)

            label_batch = [label_dict[data_line['relation']] for data_line in batch]
            all_labels += label_batch

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.LongTensor(all_labels)

    return all_features, all_labels


def label_adjust(scores, data, p=0.5):
    labels = scores.argmax(axis=-1)
    labels = labels.tolist()

    label_to_idx_score = defaultdict(list)
    for i in range(len(data)):
        label_to_idx_score[labels[i]].append([i, scores[i][labels[i]]])

    for label, idx_score in label_to_idx_score.items():
        idx_score = sorted(idx_score, key=lambda x: -x[1])
        label_to_idx_score[label] = idx_score[:int(p * len(idx_score))]

    label_to_type_count = defaultdict(Counter)
    for label, idx_score in label_to_idx_score.items():
        for idx, _ in idx_score:
            subj_type = data[idx]['subj_type']
            obj_type = data[idx]['obj_type']
            ent_type = (subj_type, obj_type)
            label_to_type_count[label][ent_type] += 1

    for label, type_count in label_to_type_count.items():
        label_to_type_count[label] = sorted(list(type_count.items()), key=lambda x: -x[1])

    label_to_type = dict()
    for label, type_count in label_to_type_count.items():
        label_to_type[label] = type_count[0][0]

    ranks = (-scores).argsort()

    adjusted_labels = []

    print(label_to_type)

    for i in range(len(labels)):

        subj_type = data[i]['subj_type']
        obj_type = data[i]['obj_type']
        ent_type = (subj_type, obj_type)

        adjusted_labels.append(labels[i])

        if label_to_type[labels[i]] != ent_type:
            rank = ranks[i]
            for label in rank:
                if label_to_type[label] == ent_type:
                    adjusted_labels[-1] = label
                    break

    return adjusted_labels


def get_top_prediction(predictions, scores, labels, p):
    label_to_idx_score = defaultdict(list)
    for i in range(len(predictions)):
        label_to_idx_score[predictions[i]].append([i, scores[i][predictions[i]]])

    for label, idx_score in label_to_idx_score.items():
        idx_score = sorted(idx_score, key=lambda x: -x[1])
        label_to_idx_score[label] = idx_score[:int(p * len(idx_score))]

    idx_list = []

    for idx_score in label_to_idx_score.values():
        idx = [ele[0] for ele in idx_score]
        idx_list += idx

    idx_list = sorted(idx_list)

    predictions = [predictions[idx] for idx in idx_list]
    labels = [labels[idx] for idx in idx_list]

    return idx_list, predictions, labels


def load_data(path):
    data = []
    with open(path) as f:
        for line in f:
            line = json.loads(line)
            data.append(line)

    label_dict = dict()
    for line in data:
        if line['raw_data']['relation'] not in label_dict:
            label_dict[line['raw_data']['relation']] = len(label_dict)

    feature_data = []

    for line in data:
        tokens = line['raw_data']['raw_data']['token']
        relation = line['raw_data']['relation']
        subj_type = line['raw_data']['raw_data']['h']['type']  # line['raw_data']['raw_data']['subj_type']#
        obj_type = line['raw_data']['raw_data']['t']['type']  # line['raw_data']['raw_data']['obj_type']#
        in_sent_words = line['in_sent_prediction']
        all_prediction = line['prediction']
        feature_data.append(
            {'tokens': tokens, 'subj_type': subj_type, 'obj_type': obj_type, 'in_sent_words': in_sent_words,
             'all_pred': all_prediction, 'relation': relation})

    return data, feature_data, label_dict


def main():
    data, feature_data, label_dict = load_data(ARGS.data_path)
    in_sent_features, labels = get_uctopic_features(feature_data, label_dict)
    all_pred_features, _ = get_uctopic_features_all_pred(feature_data, label_dict)
    features = torch.cat([in_sent_features, all_pred_features], dim=-1)
    score_cosine = get_GMM(features, labels, ARGS.num_classes, ARGS)

    predictions = label_adjust(score_cosine, feature_data, ARGS.p)
    preds_remapped = predictions

    id2label = dict()
    for label, idx in label_dict.items():
        id2label[idx] = label

    with open(ARGS.output_path, 'w') as fout:

        for data_line, pred, scores in zip(data, preds_remapped, score_cosine):
            raw_data = data_line['raw_data']
            raw_data['cluster_pred'] = int(pred)  # id2label[pred]
            raw_data['cluster_score'] = float(scores[int(pred)])

            fout.write(json.dumps(raw_data) + '\n')


if __name__ == '__main__':
    main()
