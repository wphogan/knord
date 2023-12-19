<div align="center">

# Open-world Semi-supervised Generalized Relation Discovery Aligned in a Real-world Setting

</div>

## Description

This repo contains the source code for the
paper [Open-world Semi-supervised Generalized Relation Discovery Aligned in a Real-world Setting](https://aclanthology.org/2023.emnlp-main.880/).

## Project Structure

```
┌──/baselines/         ⭠ Code for baseline models
│
├──/clustering/        ⭠ Code for training prompt model, clustering, and generating weak labels
│
├──/configs/           ⭠ Configuration files
│
├──/data/              ⭠ Project data (coming soon)
│
├──/logs               ⭠ Logs and saved checkpoints (coming soon)
│
├──/model/             ⭠ Model code
│
├──/preprocess/        ⭠ Preprocessing scripts
│
├──/saved_models/      ⭠ Saved models (coming soon)
│
├──/utils/             ⭠ Utilities
│ 
├── requirements.txt   ⭠ File for installing Python dependencies
├── run_knord.py       ⭠ Script to run KNoRD model
├── ...                ...        
├── run_{BASELINE}.py  ⭠ Scripts to run {BASELINE} model
└── ...                ...      
```

## Instructions

### 📓 Pre-run notes:
- For convenience, we provide the [preprocessed datasets and saved models](https://drive.google.com/drive/folders/1bQo-5A96Qjj4TvtVtDfO6ogMqDALXGfh?usp=drive_link) for our main experiments.
- By using the preprocessed data and saved models, you can jump to any stage detailed below without needing to run the
  previous stages.
- Please see the [KNoRD paper](https://aclanthology.org/2023.emnlp-main.880/) (link coming soon) for details about each stage of training and evaluation.
- The code in this repo was developed using Python (v3.9), PyTorch (v1.12.1), Hugging Face transformers (v4.22.2), and
  CUDA (v11.6)
- Run settings are specified in the configuration files in `/configs`. Default settings are set in `/utils/utils_config.py`. All settings can be overridden via the command
  line using the `--` prefix. For example, to override the `batch_size` setting in `/configs/knord.yaml`,
  run: `python run_knord.py --config=knord.yaml --batch_size=16`.

### 0️⃣ Initialize
- Install dependencies from `requirements.txt`
- Download [data and saved KNoRD models](https://drive.google.com/drive/folders/1bQo-5A96Qjj4TvtVtDfO6ogMqDALXGfh?usp=drive_link).
- Unzip `data.zip` and then move both `data` and `saved_models` into the project's root directory.
- For data preprocessing steps, see the instructions in the Appendix of the paper.

### 1️⃣ Prompt-model training, GMM clustering, and weak label generation
- Detailed instructions to train the prompt model, cluster using GMM, and generate weak labels are provided in the `/clustering` subdirectory.

### 2️⃣ Training KNoRD:
- To train a model using, edit the configuration file `/configs/knord.yaml` and run:

```
python run_knord.py --config=knord.yaml --exp_description=main_experiment
```

### 3️⃣ Evaluation:
- To evaluate models, add the '--eval_only' flag to the command above. For example: 

```
python run_knord.py --config=knord.yaml --eval_only
```

<br>
<br>
<br>

---
_If you found this code useful, please consider citing our paper:_


```
@inproceedings{hogan-etal-2023-open,
    title = "Open-world Semi-supervised Generalized Relation Discovery Aligned in a Real-world Setting",
    author = "Hogan, William  and
      Li, Jiacheng  and
      Shang, Jingbo",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.880",
    doi = "10.18653/v1/2023.emnlp-main.880",
    pages = "14227--14242",
    abstract = "Open-world Relation Extraction (OpenRE) has recently garnered significant attention. However, existing approaches tend to oversimplify the problem by assuming that all instances of unlabeled data belong to novel classes, thereby limiting the practicality of these methods. We argue that the OpenRE setting should be more aligned with the characteristics of real-world data. Specifically, we propose two key improvements: (a) unlabeled data should encompass known and novel classes, including negative instances; and (b) the set of novel classes should represent long-tail relation types. Furthermore, we observe that popular relations can often be implicitly inferred through specific patterns, while long-tail relations tend to be explicitly expressed. Motivated by these insights, we present a method called KNoRD (Known and Novel Relation Discovery), which effectively classifies explicitly and implicitly expressed relations from known and novel classes within unlabeled data. Experimental evaluations on several Open-world RE benchmarks demonstrate that KNoRD consistently outperforms other existing methods, achieving significant performance gains.",
}
```