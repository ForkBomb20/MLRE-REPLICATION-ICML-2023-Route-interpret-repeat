## Table of Contents

1. [Environment setup](#environment-setup)
2. [Downloading data](#downloading-data)
   - [(a) Downloading data](#a-downloading-vision-and-skin-data)
3. [Data preprocessing](#data-preprocessing)
   - [(a) Preprocessing CUB200](#a-preprocessing-cub200)
4. [Training pipeline](#training-pipleline)
   - [(a) Running MoIE](#a-running-moie)
   - [(b) Compute the performance metrics](#b-compute-the-performance-metrics)
   - [(c) Validating the concept importance](#c-validating-the-concept-importance)
5. [Generated Local Explanations](#generated-local-explanations)
6. [Suggestions](#suggestions)
7. [Checkpoints](#checkpoints)

## Environment setup

```bash
conda env create --name python_3_7_rtx_6000 -f environment.yml
conda activate python_3_7_rtx_6000
```

## Downloading data

After downloading data from the below links, search for `--data-root` variable in the codebase and replace the
appropriate paths for all the different datasets. Also search for `/ocean/projects/asc170022p/shg121/PhD/ICLR-2022`
and replace with appropriate paths.

### (a) Downloading data

| Dataset | Description                 | URL                                                                       |
| ------- | --------------------------- | ------------------------------------------------------------------------- | --- |
| CUB-200 | Bird Classification dataset | [CUB-200 Official](https://www.vision.caltech.edu/datasets/cub_200_2011/) |     |

## Data preprocessing

### (a) Preprocessing CUB200

To get the CUB200 metadata and dataset splits
follow [Logic Explained network](https://github.com/pietrobarbiero/logic_explained_networks/tree/master/data).
Once the json files are downloaded, search for `--json-root` variable in the codebase and replace the
appropriate paths for all the different datasets.

To preprocess the concepts for CUB200, follow:

```python
python ./src/codebase/data_preprocessing/download_cub.py
```

## Training pipeline

All the scripts for training MoIE, is included in [`./src/scripts`](/src/scripts) folder for all the datasets and
architectures with comments. Follow every command sequentially of each script to train/test the Blackbox (BB), concept
predictor (t), explainers (g) and residuals (r).

- As a first step find and replace the project path `/ocean/projects/asc170022p/shg121/PhD/ICLR-2022` from the whole
  codebase with appropriate path.

- Also, after training and testing MoIE, as the last step in each
  script, [`FOLs_vision_main.py`](/src/codebase/FOLs_vision_main.py) file is responsible for generating instance
  specific FOL. This file
  uses [`./src/codebase/Completeness_and_interventions/paths_MoIE.json`](/src/codebase/Completeness_and_interventions/paths_MoIE.json)
  file where we keep all the paths and filenames of the checkpoints of Blackbox (bb), concept predictor (t), explainer (
  g), and residual (r). Replace those paths and filenames with the appropriate ones based on the experiments. Refer
  below for the description of the
  variables [`paths_MoIE.json`](/src/codebase/Completeness_and_interventions/paths_MoIE.json):

| Variable        | Description                                                          |
| --------------- | -------------------------------------------------------------------- |
| `cub_ResNet101` | Root variable for CUB200 dataset with Resnet101 as the Blackbox (BB) |

- Note the root follow dataset_BB_architecture format. **Do not modify this format**. For each of the above
  roots [`paths_MoIE.json`](/src/codebase/Completeness_and_interventions/paths_MoIE.json) file, based on the dataset and
  architectures, edit the values in `MoIE_paths`, `t`
  , `bb` with appropriate checkpoint paths and files for the different experts (g), concept predictors (t) and
  Blackbox (
  bb).

**Refer to the following sections for details of each of the scripts.**

### (a) Running MoIE

| Script name                                                 | Description                                                   | Comment                                                                                                                      |
| ----------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| [`./src/scripts/cub_resnet.sh`](/src/scripts/cub_resnet.sh) | Script for CUB200 dataset with Resnet101 as the Blackbox (BB) | Included train/test and FOL generation script for the Blackbox (BB), concept predictor (t), explainers (g) and residuals (r) |

Reference

- [ResNet-101 on CUB-200](https://github.com/zhangyongshun/resnet_finetune_cub)

### (b) Compute the performance metrics

To compute performance metrics (accuracy/AUROC) for all the experts cumulatively (Table 2 in the paper), refer below

| Notebook                                                                                                                          | Description                                                   |
| --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| [`./src/codebase/iPython/Cumulative_performance/CUB-Resnet.ipynb`](/src/codebase/iPython/Cumulative_performance/CUB-Resnet.ipynb) | Script for CUB200 dataset with Resnet101 as the Blackbox (BB) |

### (c) Validating the concept importance

In the paper, we validate in the importance of the extracted concepts using three experiments:

1. Zeroing out the important concepts
2. Computing the completeness scores of the important concept
   - Before running the script for completeness score, run the following scripts to create the dataset to train the
     projection model in completeness score paper:

| Notebook                                                                                                                      | Description                                                   |
| ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| [`./src/codebase/iPython/Completeness_dataset/CUB_Resnet.ipynb`](/src/codebase/iPython/Completeness_dataset/CUB_Resnet.ipynb) | Script for CUB200 dataset with Resnet101 as the Blackbox (BB) |

3. Performing test time interventions

Please refer to the table below for the scripts to replicate the above experiments (zeroing out the concepts,
completeness scores and test time interventions):

| Scripts                                                                       | Description                                                                                                                                       |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`./src/scripts/zero_out_concepts.sh`](/src/scripts/zero_out_concepts.sh)     | Script to zero out the important concepts                                                                                                         |
| [`./src/scripts/completeness_scores.sh`](/src/scripts/completeness_scores.sh) | Script to estimate the completeness scores of the important concepts                                                                              |
| [`./src/scripts/tti.sh`](/src/scripts/tti.sh)                                 | Script to perform test time interventions for the important concepts                                                                              |
| [`./src/codebase/tti_experts.sh`](/src/scripts/tti_experts.sh)                | Script to perform test time interventions for the important concepts corresponding to only the **harder** samples covered by the last two experts |

## Generated Local Explanations

We have included the instance-specific explanations per expert for each dataset in the folder [`./explanations`](/explanations).

## Suggestions

Most of the _argparse_ variables are self-explanatory. However, in order to perform the experiments successfully, give
the correct paths and files to the following variables in `train_explainer_<dataset>.py` and `test_explainer_<dataset>.py`.

- For `train_explainer_<dataset>.py` (ex. [`train_explainer_CUB.py`](src/codebase/train_explainer_CUB.py)), follow the rules:

  1. `--checkpoint-model` : Don't include this variable for the 1st iteration. For 2nd iteration and onwards, include
     the checkpoint files of all the experts of **previous iterations while training for the expert (
     g) (`--expert-to-train "explainer"`)**. For example: if the current iteration is 3, include the checkpoint files
     for the expert 1 and expert 2 sequentially. While **training the residual (`--expert-to-train "residual"`)**,
     include the checkpoint files of all the experts **including the current iteration**.
  2. `--checkpoint-residual` : Don't include this variable for the 1st iteration. For 2nd iteration and onwards,
     include the checkpoint files of all the residuals of **previous iterations** while training the expert (
     g) (`--expert-to-train "explainer"`) and the residual (`--expert-to-train "explainer"`). For example: if the
     current iteration is 3, include the checkpoint files for the residual 1 and residual 2 sequentially.
  3. `--prev_explainer_chk_pt_folder` : Don't include this variable for the 1st iteration. For 2nd iteration and
     onwards, include the folders of the checkpoint files of all the experts of **previous iterations**. For example:
     if the current iteration is 3, include the checkpoint folders for the expert 1 and expert 2 sequentially. For all
     the datasets other than MIMIC-CXR, include the absolute path. For MIMIC-CXR, only include the experiment folder
     where the checkpoint file will be stored.

  Refer to the following example command for the 3rd iteration for CUB200 dataset with VIT as the blackbox to train the
  expert:

  ```python
  python ./src/codebase/train_explainer_CUB.py --expert-to-train "explainer" --checkpoint-model checkpt_expert1 checkpt_expert2 --checkpoint-residual checkpt_residual1 checkpt_residual2 --prev_explainer_chk_pt_folder checkpt_folder_exper1 checkpt_folder_expert2 --root-bb "lr_0.03_epochs_95" --checkpoint-bb "VIT_CUBS_8000_checkpoint.bin" --iter 3 --dataset "cub" --cov cov_iter1 cov_iter2 cov_iter3 --bs 16 --dataset-folder-concepts "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr learning_rate_iter1 learning_rate_iter2 learning_rate_iter3 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "VIT" --arch "VIT-B_16"
  ```

  Similarly, refer to the following example command for the 3rd iteration for **CUB200** dataset with **VIT** as the
  blackbox to train the residual:

  ```python
  python ./src/codebase/train_explainer_CUB.py --expert-to-train "residual" --checkpoint-model checkpt_expert1 checkpt_expert2 checkpt_expert3 --checkpoint-residual checkpt_residual1 checkpt_residual2 --prev_explainer_chk_pt_folder checkpt_folder_exper1 checkpt_folder_expert2 --root-bb "lr_0.03_epochs_95" --checkpoint-bb "VIT_CUBS_8000_checkpoint.bin" --iter 3 --dataset "cub" --cov cov_iter1 cov_iter2 cov_iter3 --bs 16 --dataset-folder-concepts "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr learning_rate_iter1 learning_rate_iter2 learning_rate_iter3 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "VIT" --arch "VIT-B_16"
  ```

- For `test_explainer_<dataset>.py` (ex. [`test_explainer_CUB.py`](src/codebase/test_explainer_CUB.py)
  , [`test_explainer_ham10k.py`](src/codebase/test_explainer_ham10k.py) etc.), follow the rules:

  1. `--checkpoint-model` : Don't include this variable for the 1st iteration. For 2nd iteration and onwards, include
     the checkpoint files of all the experts **including the current iteration** while testing the expert (
     g) (`--expert-to-train "explainer"`) and the residual (`--expert-to-train "explainer"`).
  2. `--checkpoint-residual` : Don't include this variable for the 1st iteration. For 2nd iteration and onwards,
     include the checkpoint files of all the residuals of **previous iterations** while training for the expert (
     g) (`--expert-to-train "explainer"`)**. For example: if the current iteration is 3, include the checkpoint files
     for the residual 1 and residual 2 sequentially. While **testing the residual (`--expert-to-train "residual"`)**,
     include the checkpoint files of all the residuals **including the current iteration\*\*.
  3. `--prev_explainer_chk_pt_folder` : Don't include this variable for the 1st iteration. For 2nd iteration and
     onwards, include the folders of the checkpoint files all the experts of **previous iterations**. For example: if
     the current iteration is 3, include the checkpoint folders for the expert 1 and expert 2 sequentially. For all
     the datasets other than MIMIC-CXR, include the absolute path. For MIMIC-CXR, only include the experiment folder
     where the checkpoint file will be stored.

  Refer to the following example command for the 3rd iteration for **CUB200** dataset with **VIT** as the blackbox to
  test the expert:

  ```python
  python ./src/codebase/test_explainer_CUB.py --expert-to-train "explainer" --checkpoint-model checkpt_expert1 checkpt_expert2 checkpt_expert3 --checkpoint-residual checkpt_residual1 checkpt_residual2 --prev_explainer_chk_pt_folder checkpt_folder_exper1 checkpt_folder_expert2 --root-bb "lr_0.03_epochs_95" --checkpoint-bb "VIT_CUBS_8000_checkpoint.bin" --iter 3 --dataset "cub" --cov cov_iter1 cov_iter2 cov_iter3 --bs 16 --dataset-folder-concepts "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr learning_rate_iter1 learning_rate_iter2 learning_rate_iter3 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "VIT" --arch "VIT-B_16"
  ```

  Similarly, refer to the following example command for the 3rd iteration for CUB200 dataset with VIT as the blackbox to
  test the residual:

  ```python
  python ./src/codebase/test_explainer_CUB.py --expert-to-train "residual" --checkpoint-model checkpt_expert1 checkpt_expert2 checkpt_expert3 --checkpoint-residual checkpt_residual1 checkpt_residual2 checkpt_residual3 --prev_explainer_chk_pt_folder checkpt_folder_exper1 checkpt_folder_expert2 --root-bb "lr_0.03_epochs_95" --checkpoint-bb "VIT_CUBS_8000_checkpoint.bin" --iter 3 --dataset "cub" --cov cov_iter1 cov_iter2 cov_iter3 --bs 16 --dataset-folder-concepts "lr_0.03_epochs_95_ViT-B_16_layer4_VIT_sgd_BCE" --lr learning_rate_iter1 learning_rate_iter2 learning_rate_iter3 --input-size-pi 2048 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --layer "VIT" --arch "VIT-B_16"
  ```

Also make sure the following variables are correct:

- `--cov`: Coverages of each iteration separated by a space as in the above commands.
- `--lr`: Learning rates of each expert separated by a space as in the above commands.
- `--data-root`: Dataset path of images, labels and concepts (if exists)
- `--logs`: Path of tensorboard logs

## Checkpoints

For the checkpoints of the pretrained blackboxes and concept banks, refer below:

| Blackbox                                                                                   | Concept predictor (t) / Concept banks                                                      |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| [CUB200-VIT](https://drive.google.com/drive/u/1/folders/1nDmJklw3UJy_75Oh23BvzCw6VkGFWet1) | [CUB200-VIT](https://drive.google.com/drive/u/1/folders/1fSI231IcaClK6OAZrIg6ptVXRaeIpGkh) |

Note for HAM10k, we add the extracted concept bank after training `t`. No need to train t for HAM10k and SIIM-ISIC, if
this concept bank is used. For others, the above paths contain the checkpoints of `t`. Use these checkpoints to extract
the concepts.
