<div align="center">

# Open-world Semi-supervised Generalized Relation Discovery Aligned in a Real-world Setting

</div>

## Description

This repo contains the source code for the
paper [Open-world Semi-supervised Generalized Relation Discovery Aligned in a Real-world Setting](https://paper_coming_soon).

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
- For convenience, we provide the
  complete [preprocessed datasets, splits, saved models, and logs](https://data_coming_soon) (link coming soon) for all of our experiments.
- By using the preprocessed data and saved models, you can jump to any stage detailed below without needing to run the
  previous stages.
- Please see the [KNoRD paper](https://paper_coming_soon) (link coming soon) for details about each stage of training and evaluation.
- The code in this repo was developed using Python (v3.9), PyTorch (v1.12.1), Hugginface transformers (v4.22.2), and
  CUDA (v11.6)
- Run settings are specified in the configuration files in `/configs`. Default settings are set in `/utils/utils_config.py`. All settings can be overridden via the command
  line using the `--` prefix. For example, to override the `batch_size` setting in `/configs/knord.yaml`,
  run: `python run_knord.py --config=knord.yaml --batch_size=16`.

### 0️⃣ Initialize
- Install dependencies from `requirements.txt`
- Download [data and saved KNoRD models](https://data_coming_soon) (link coming soon).
- Unzip `data.zip` and then move both `data` and `saved_models` into the project's root directory.
- [OPTIONAL] All the preprocessed data is provided, but if you'd like to preprocess the data yourself, follow the data
  preprocessing instructions in the Appendix of the paper.

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
@inproceedings{
  coming-soon
} 
```