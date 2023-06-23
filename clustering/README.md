## Prompting Model for KNoRD

We include a simple introduction about how to use the prompting model to get weak supervisions for the KNoRD model.
We take FewRel dataset without `no_relation` classes as an example in this repo.

### Data Preparation

To prepare the data, we need to have a labeled dataset (`fewrel_label_0.15_nonorel.jsonl`), an unlabeled dataset (`fewrel_unlabel_0.15_nonorel.jsonl`), a relation-id dictionary (`rel2id.json`), and a relation-name dictionary (`rel2name.json`).

### Train a prompting model
Run the following command to train a prompting model:
```shell
python train.py \
    --dataset_dir ./data/fewrel_no_norel \
    --label_data fewrel_label_0.15_nonorel.jsonl \
    --batch_size 32 \
    --output_dir fewrel_no_norel_results \
    --mlm_probability 0.15 \
    --device cuda:3
```

### Infer on unlabeled dataset
Run the following command to infer on unlabeled dataset:
```shell
python test.py \
    --model_name ./fewrel_no_norel_results \
    --dataset_dir ./data/fewrel_no_norel \
    --label_data fewrel_label_0.15_nonorel.jsonl \
    --unlabel_data fewrel_unlabel_0.15_nonorel.jsonl \
    --output_dir fewrel_no_norel_results \
    --batch_size 32 \
    --device cuda:1
```

### Get weak supervisions by clustering
Run the following command to get weak supervisions by clustering:
```shell
python clustering.py \
    --output_path fewrel_with_norel_results/cluster_results.json \
    --data_path fewrel_with_norel_results/test_result_train+test.json \
    --batch_size 32 \
    --num_classes 120 \
    --gpu 3
```
where `--num_classes` is the number of relations (provided by the user) in the dataset.

The output of clustering will include all data and corresponding weak labels for the second stage of training.