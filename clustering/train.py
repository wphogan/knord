import argparse
import json
import logging
import os
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForMaskedLM, get_scheduler, SchedulerType

from clustering.data import REDataset

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="OWRelation")
    parser.add_argument(
        "--model_name",
        type=str,
        default='roberta-base',
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
        "--label_data",
        type=str,
        default=None,
        help="Labeled data.",
    )
    parser.add_argument(
        "--unlabel_data",
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
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
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
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Total number of training epochs to perform.")
    parser.add_argument("--num_warmup_steps", type=int, default=0,
                        help="Number of steps for the warmup in the lr scheduler.")
    args = parser.parse_args()

    return args


def train_dev_split_by_relation(data, dev_relation_num, exclude_relations=['no_relation']):
    exlcude_set = set(exclude_relations)
    candidate_set = set()
    for line in data:
        if line['relation'] not in exlcude_set:
            candidate_set.add(line['relation'])
    candidates = list(candidate_set)
    dev_relations = set(random.sample(candidates, k=dev_relation_num))

    train_data = []
    dev_data = []

    for line in data:
        if line['relation'] in dev_relations:
            dev_data.append(line)
        else:
            train_data.append(line)

    return train_data, dev_data


def main():
    args = parse_args()

    logging.basicConfig(
        filename='logging.txt',
        filemode='a',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = RobertaForMaskedLM.from_pretrained(args.model_name)
    model.to(args.device)

    special_list = ["[object_start]", "[object_end]", "[subject_start]", "[subject_end]"]
    tokenizer.add_special_tokens({'additional_special_tokens': special_list})
    model.resize_token_embeddings(len(tokenizer))

    data = []
    with open(os.path.join(args.dataset_dir, args.label_data)) as f:
        for line in f:
            data.append(json.loads(line))

    train_data, eval_data = train_dev_split_by_relation(data, dev_relation_num=5)

    if args.label_data.startswith('fewrel'):
        with open(os.path.join(args.dataset_dir, 'rel2name.json')) as f:
            rel2name = json.load(f)

        for line in train_data:
            line['relation'] = rel2name[line['relation']][0]

        for line in eval_data:
            line['relation'] = rel2name[line['relation']][0]

    train_data = REDataset(train_data, tokenizer, args)
    eval_data = REDataset(eval_data, tokenizer, args, is_eval=True)

    train_dataloader = DataLoader(train_data, shuffle=True, collate_fn=train_data.collate_fn,
                                  batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_data, collate_fn=eval_data.collate_fn, batch_size=args.batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    num_update_steps_per_epoch = len(train_dataloader)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  mlm_probability = {args.mlm_probability}")

    progress_bar = tqdm(range(args.max_train_steps))

    best_loss = float('inf')
    patient = 3

    global_step = 0
    for epoch in range(args.num_epochs):

        model.train()

        for batch in train_dataloader:

            if global_step % 50 == 0:

                model.eval()
                losses = []
                for batch in eval_dataloader:

                    for k, v in batch.items():
                        batch[k] = v.to(args.device)

                    with torch.no_grad():
                        outputs = model(**batch)

                    loss = outputs.loss
                    losses.append(float(loss.detach().cpu()))

                eval_loss = sum(losses) / len(losses)

                logger.info(f"epoch {epoch}: Eval loss: {eval_loss}")

                if eval_loss < best_loss:

                    logger.info("Best loss on eval data. Save checkpoint.")
                    best_loss = eval_loss
                    patient = 3
                    model.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)

                else:

                    patient -= 1

                    if patient == 0:
                        logger.info("Patient becomes 0. Stop training.")
                        exit(0)

                model.train()

            for k, v in batch.items():
                batch[k] = v.to(args.device)

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1


if __name__ == "__main__":
    main()
