## Table of Contents

1. [Environment setup](#1-environment-setup)
2. [Downloading data](#2-downloading-data)
3. [Data preprocessing](#3-data-preprocessing-cub200)
4. [Training pipeline](#4-training-pipeline)
5. [Script details](#5-script-details)
   - [(a) Running MoIE](#a-running-moie)
   - [(b) Computing performance metrics](#b-computing-performance-metrics)
   - [(c) Validating concept importance](#c-validating-concept-importance---3-experiments)
6. [Generated Local Explanations](#6-generated-local-explanations)
7. [Train/Test Arguments](#7-traintest-arguments)
   - 7-1. [Arguments Example](#7-1-arguments-example)
8. [Checkpoints](#8-checkpoints)

# Accessing Great Lakes
```bash
uniqname@greatlakes.arc-ts.umich.edu
cd /scratch/eecs498f25s00{7 or 8}_class_root/eecs498f25s00{7 or 8}_class/
cd uniqname
module load python/3.13.2
```

## Runing on an interactive node:
```bash
salloc --partition=gpu --gres=gpu:1 --time=8:00:00 --mem=64G --account=eecs498f25s00{7 or 8}_class
module load python/3.13.2
source env/bin/activate
./src/scrpts/transfer_to_tmp.sh
```

Once you're done with the interactive node:

```bash
scancel $SLURM_JOB_ID
```

## 1. Environment setup
Python version 3.13
```bash
python3 -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
```

## 2. Downloading data

[CUB-200 Official](https://www.vision.caltech.edu/datasets/cub_200_2011/)

Download CUB_200_2011.tgz (1.2GB)

## 3. Data preprocessing CUB200

To preprocess the concepts for CUB200, move the **inner** `CUB_200_2011` folder along with `attributes.txt` from the extracted `CUB_200_2011` folder into the root directory of the repository. Also ensure that a `data/` folder exists in the root directory. Then, from the root directory, run:
```python
python ./src/codebase/data_preprocessing/download_cub.py
```

Once json files are downloaded, search for `--json-root` and `--data-root` variable in this codebase and replace the appropriate paths.

## 4. Training pipeline

All scripts for training MoIE is in [`./src/scripts`](/src/scripts).
Follow every command sequentially of each script to train/test the Blackbox (BB), concept predictor (t), explainers (g) and residuals (r).

- First, find and replace the project path `.` from the whole codebase with appropriate path.

- After training and testing MoIE, as last step in each script, [`FOLs_vision_main.py`](/src/codebase/FOLs_vision_main.py) file is responsible for generating instance specific FOL. This file uses [`./src/codebase/Completeness_and_interventions/paths_MoIE.json`](/src/codebase/Completeness_and_interventions/paths_MoIE.json) file where we keep all the paths and filenames of the checkpoints of Blackbox (bb), concept predictor (t), explainer (g), and residual (r). Replace those paths and filenames with the appropriate ones.

- **Do not modify format of** [`paths_MoIE.json`](/src/codebase/Completeness_and_interventions/paths_MoIE.json) file, edit values `MoIE_paths`, `t`, `bb` with appropriate checkpoint paths and files for the different experts (g), concept predictors (t) and Blackbox (bb).

## 5. Script details

### (a) Running MoIE

[`./src/scripts/cub_resnet.sh`](/src/scripts/cub_resnet.sh): Included train/test and FOL generation script for the Blackbox (BB), concept predictor (t), explainers (g) and residuals (r)

Reference: [ResNet-101 on CUB-200](https://github.com/zhangyongshun/resnet_finetune_cub)

### (b) Computing performance metrics

To compute performance metrics (accuracy/AUROC) for all the experts cumulatively (Table 2 in the paper), refer [`./src/codebase/iPython/Cumulative_performance/CUB-Resnet.ipynb`](/src/codebase/iPython/Cumulative_performance/CUB-Resnet.ipynb)

### (c) Validating concept importance - 3 experiments

1. [`./src/scripts/zero_out_concepts.sh`](/src/scripts/zero_out_concepts.sh) - Zeroing out the important concepts
2. [`./src/scripts/completeness_scores.sh`](/src/scripts/completeness_scores.sh) - Computing the completeness scores of the important concepts
   - Run [`./src/codebase/iPython/Completeness_dataset/CUB_Resnet.ipynb`](/src/codebase/iPython/Completeness_dataset/CUB_Resnet.ipynb) first to create the dataset to train the projection model in completeness score paper.
3. [`./src/scripts/tti.sh`](/src/scripts/tti.sh) - Performing test time interventions of important concepts
   - [`./src/codebase/tti_experts.sh`](/src/scripts/tti_experts.sh) - Perform test time interventions for only the **harder** samples covered by last two experts

## 6. Generated Local Explanations

Instance-specific explanations per expert is in [`./explanations`](/explanations).

## 7. Train/Test Arguments

Make sure following variables are correct:

- `--cov`: Coverages of each iteration separated by a space as in the above commands.
- `--lr`: Learning rates of each expert separated by a space as in the above commands.
- `--data-root`: Dataset path of images, labels and concepts (if exists)
- `--logs`: Path of tensorboard logs

To perform the experiments successfully, give the correct paths and files to the following variables

- in [`train_explainer_CUB.py`](src/codebase/train_explainer_CUB.py):

  1. `--checkpoint-model` : Don't include for 1st iteration. Starting 2nd iteration, include checkpoint files of all the experts of **previous iterations** while training for the expert (g) (`--expert-to-train "explainer"`).

     - Ex. if current iteration is 3, include checkpoint files for expert 1 and 2 sequentially. While **training the residual** (`--expert-to-train "residual"`), include checkpoint files of all experts **including current iteration**.

  2. `--checkpoint-residual` : Don't include for 1st iteration. Starting 2nd iteration, include checkpoint files of all residuals of **previous iterations** while training the expert (g) and the residual.

     - Ex. if current iteration is 3, include checkpoint files for residual 1 and 2 sequentially.

  3. `--prev_explainer_chk_pt_folder` : Don't include for 1st iteration. Starting 2nd iteration, include folders of all expert checkpoint files of **previous iterations**.

     - Ex. if current iteration is 3, include checkpoint folders for expert 1 and 2 sequentially. Include absolute path for CUB dataset.

- in [`test_explainer_CUB.py`](src/codebase/test_explainer_CUB.py), do the following points differently:

  1. `--checkpoint-model` : **including current iteration** while testing expert and residual

  2. `--checkpoint-residual` : Only when **testing the residual**, include checkpoint files of all residuals **including the current iteration**.

  3. `--prev_explainer_chk_pt_folder` : No change

## 7-1. Arguments example

For example setting = 3rd iteration for CUB200+VIT blackbox

Base arguments for all example commands:

```python
  --prev_explainer_chk_pt_folder checkpt_folder_expert1 checkpt_folder_expert2
  --root-bb "lr_0.03_epochs_95"
  --checkpoint-bb "VIT_CUBS_8000_checkpoint.bin"
  --iter 3
  --dataset "cub"
  --cov cov_iter1 cov_iter2 cov_iter3
  --bs 16
  --dataset-folder-concepts "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE"
  --lr learning_rate_iter1 learning_rate_iter2 learning_rate_iter3
  --input-size-pi 2048
  --temperature-lens 0.7
  --lambda-lens 0.0001
  --alpha-KD 0.9
  --temperature-KD 10
  --hidden-nodes 10
  --layer "VIT"
  --arch "VIT-B_16"
```

Train expert: (explainer, 2, 2)

```python
python ./src/codebase/train_explainer_CUB.py
  --expert-to-train "explainer"
  --checkpoint-model checkpt_expert1 checkpt_expert2
  --checkpoint-residual checkpt_residual1 checkpt_residual2
```

Train residual: (residual, 3, 2)

```python
python ./src/codebase/train_explainer_CUB.py
  --expert-to-train "residual"
  --checkpoint-model checkpt_expert1 checkpt_expert2 checkpt_expert3
  ...
```

Test expert: (explainer, 3, 2)

```python
python ./src/codebase/test_explainer_CUB.py
  ...
  --checkpoint-model checkpt_expert1 checkpt_expert2 checkpt_expert3
  ...
```

Test residual: (residual, 3, 3)

```python
python ./src/codebase/test_explainer_CUB.py
  --expert-to-train "residual"
  --checkpoint-model checkpt_expert1 checkpt_expert2 checkpt_expert3
  --checkpoint-residual checkpt_residual1 checkpt_residual2 checkpt_residual3
```

## 8. Checkpoints

**Note: all links on original github are 404 errors**
Checkpoints of pretrained blackboxes and concept banks:

| Blackbox                                                                                   | Concept predictor (t) / Concept banks                                                      |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| [CUB200-VIT](https://drive.google.com/drive/u/1/folders/1nDmJklw3UJy_75Oh23BvzCw6VkGFWet1) | [CUB200-VIT](https://drive.google.com/drive/u/1/folders/1fSI231IcaClK6OAZrIg6ptVXRaeIpGkh) |
