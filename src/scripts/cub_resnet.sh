#!/bin/sh
#SBATCH --job-name=cub_resnet
#SBATCH --output=slurm_outs/cub_resnet_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1           # Request 5 GPUs
#SBATCH --nodes=1              # Run on one node
#SBATCH --ntasks-per-node=1    # One process per node
#SBATCH --cpus-per-task=8      # Number of CPU cores per GPU task (adjust as needed)
#SBATCH --mem=32G              # Total memory per node (adjust as needed)
#SBATCH --time=2:00:00      # Job time limit (2 hours)

pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT

slurm_output_bb_train=./out/cub_resnet_bb_train_$CURRENT.out
slurm_output_bb_test=./out/cub_resnet_bb_test_$CURRENT.out
slurm_output_t_train=./out/cub_resnet_t_train_$CURRENT.out
slurm_output_t_test=./out/cub_resnet_t_test_$CURRENT.out

slurm_output_iter1_g_train=./out/cub_resnet_iter1_g_train_$CURRENT.out
slurm_output_iter1_g_test=./out/cub_resnet_iter1_g_test_$CURRENT.out
slurm_output_iter1_residual_train=./out/cub_resnet_iter1_residual_train_$CURRENT.out

slurm_output_iter2_g_train=./out/cub_resnet_iter2_g_train_$CURRENT.out
slurm_output_iter2_g_test=./out/cub_resnet_iter2_g_test_$CURRENT.out
slurm_output_iter2_residual_train=./out/cub_resnet_iter2_residual_train_$CURRENT.out

slurm_output_iter3_g_train=./out/cub_resnet_iter3_g_train_$CURRENT.out
slurm_output_iter3_g_test=./out/cub_resnet_iter3_g_test_$CURRENT.out
slurm_output_iter3_residual_train=./out/cub_resnet_iter3_residual_train_$CURRENT.out

slurm_output_iter4_g_train=./out/cub_resnet_iter4_g_train_$CURRENT.out
slurm_output_iter4_g_test=./out/cub_resnet_iter4_g_test_$CURRENT.out
slurm_output_iter4_residual_train=./out/cub_resnet_iter4_residual_train_$CURRENT.out

slurm_output_iter5_g_train=./out/cub_resnet_iter5_g_train_$CURRENT.out
slurm_output_iter5_g_test=./out/cub_resnet_iter5_g_test_$CURRENT.out
slurm_output_iter5_residual_train=./out/cub_resnet_iter5_residual_train_$CURRENT.out

slurm_output_iter6_g_train=./out/cub_resnet_iter6_g_train_$CURRENT.out
slurm_output_iter6_g_test=./out/cub_resnet_iter6_g_test_$CURRENT.out
slurm_output_iter6_residual_train=./out/cub_resnet_iter6_residual_train_$CURRENT.out
slurm_output_iter6_residual_test=./out/cub_resnet_iter6_residual_test_$CURRENT.out

slurm_explanations=./out/cub_resnet_explanations_$CURRENT.out

# Uncomment for great lakes
module load python/3.13.2
echo "CUB-200 ResNet101"
source ./env/bin/activate
which python

# if [ ! -d /tmp/$USER/data_cub ]; then
#     echo "Copying dataset to local /tmp..."
#     mkdir -p /tmp/$USER/data_cub
#     rsync -ah --info=progress2 --ignore-existing ./data/ /tmp/$USER/data_cub
#     echo "Dataset copied to local /tmp."
# else
#     echo "Using cached local dataset."
# fi

# BB model
# BB Training scripts

# python ./src/codebase/train_BB_CUB.py \
#     --bs 16 \
#     --arch "ResNet101" \
#     --data-root "/tmp/$USER/data_cub/data/CUB_200_2011" \
#     > $slurm_output_bb_train


# BB Testing scripts
# Update ./src/codebase/Completeness_and_interventions/paths_MoIE.json file with appropriate paths for the checkpoints and outputs
python ./src/codebase/test_BB_CUB.py \
    --checkpoint-file "best_model.pth.tar" \
    --save-activations True \
    --layer "layer4" \
    --bs 16 \
    --arch "ResNet101" \
    --data-root "/tmp/$USER/data_cub/CUB_200_2011" \
    > $slurm_output_bb_test


# T model
# train
python ./src/codebase/train_t_CUB.py \
    --checkpoint-file "best_model_epoch_63.pth.tar" \
    --bs 32 \
    --layer "layer4" \
    --flattening-type "adaptive" \
    --arch "ResNet101" \
    > $slurm_output_t_train

# Test
python ./src/codebase/test_t_CUB.py \
    --checkpoint-file "best_model_epoch_63.pth.tar" \
    --checkpoint-file-t "best_model_epoch_62.pth.tar" \
    --save-concepts True \
    --bs 16 \
    --solver-LR "sgd" \
    --loss-LR "BCE" \
    --layer "layer4" \
    --flattening-type "adaptive" \
    --arch "ResNet101" \
    > $slurm_output_t_test


# MoIE Training scripts

# Common args for all explainer/residual train/test calls
common_args='
--root-bb lr_0.001_epochs_95
--checkpoint-bb best_model_epoch_63.pth.tar
--dataset cub
--bs 16
--dataset-folder-concepts lr_0.001_epochs_95_ResNet101_layer4_adaptive_sgd_BCE
--input-size-pi 2048
--temperature-lens 0.7
--lambda-lens 0.0001
--alpha-KD 0.9
--temperature-KD 10
--hidden-nodes 10
--layer layer4
--arch ResNet101
'

#---------------------------------
# iter 1
#---------------------------------
iter1_common_args='
--iter 1
--cov 0.2
--lr 0.01
'

# Train explainer
python ./src/codebase/train_explainer_CUB.py \
    --expert-to-train "explainer" \
    $iter1_common_args \
    $common_args \
    > $slurm_output_iter1_g_train

# Test explainer
python ./src/codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" \
    --expert-to-train "explainer" \
    $iter1_common_args \
    $common_args \
    >  $slurm_output_iter1_g_test

# Train residual
python ./src/codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" \
    --expert-to-train "residual" \
    $iter1_common_args \
    $common_args \
    > $slurm_output_iter1_residual_train

#---------------------------------
# iter 2
#---------------------------------
iter2_common_args='
--prev_explainer_chk_pt_folder ./checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1
--checkpoint-residual model_residual_best_model_epoch_1.pth.tar
--iter 2
--cov 0.2 0.2
--lr 0.01 0.01
'

python ./src/codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" \
    --expert-to-train "explainer" \
    $iter2_common_args \
    $common_args \
    >  $slurm_output_iter2_g_train

python ./src/codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" \
    --expert-to-train "explainer" \
    $iter2_common_args \
    $common_args \
    > $slurm_output_iter2_g_test

python ./src/codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" \
    --expert-to-train "residual" \
    $iter2_common_args \
    $common_args \
    > $slurm_output_iter2_residual_train

#---------------------------------
# iter 3
#---------------------------------
iter3_common_args='
--prev_explainer_chk_pt_folder ./checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1 ./checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2
--checkpoint-residual model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar
--iter 3
--cov 0.2 0.2 0.2
--lr 0.01 0.01 0.01
'

python ./src/codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar"  "model_g_best_model_epoch_188.pth.tar" \
    --expert-to-train "explainer" \
    $iter3_common_args \
    $common_args \
    > $slurm_output_iter3_g_train


python ./src/codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" \
    --expert-to-train "explainer" \
    $iter3_common_args \
    $common_args \
    > $slurm_output_iter3_g_test


python ./src/codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" \
    --expert-to-train "residual" \
    $iter3_common_args \
    $common_args \
    > $slurm_output_iter3_residual_train

#---------------------------------
# iter 4
#---------------------------------
iter4_common_args='
--prev_explainer_chk_pt_folder ./checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1 ./checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2 ./checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter3
--checkpoint-residual model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar
--iter 4
--cov 0.2 0.2 0.2 0.2
--lr 0.01 0.01 0.01 0.01
'
python ./src/codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" \
    --expert-to-train "explainer" \
    $iter4_common_args \
    $common_args \
    > $slurm_output_iter4_g_train


python ./src/codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" \
    --expert-to-train "explainer" \
    $iter4_common_args \
    $common_args \
    > $slurm_output_iter4_g_test


python ./src/codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" \
    --expert-to-train "residual" \
    $iter4_common_args \
    $common_args \
    > $slurm_output_iter4_residual_train

# ---------------------------------
# iter 5
# ---------------------------------
iter5_common_args='
--prev_explainer_chk_pt_folder ./checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1 ./checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2 ./checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter3 ./checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter4
--checkpoint-residual model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar
--iter 5
--cov 0.2 0.2 0.2 0.2 0.2
--lr 0.01 0.01 0.01 0.01 0.01
'
python ./src/codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar"  "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" \
    --expert-to-train "explainer" \
    $iter5_common_args \
    $common_args \
    > $slurm_output_iter5_g_train


python ./src/codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" "model_g_best_model_epoch_345.pth.tar" \
    --expert-to-train "explainer" \
    $iter5_common_args \
    $common_args \
    > $slurm_output_iter5_g_test


python ./src/codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" "model_g_best_model_epoch_345.pth.tar" \
    --expert-to-train "residual" \
    $iter5_common_args \
    $common_args \
    > $slurm_output_iter5_residual_train

# ---------------------------------
# iter 6
# ---------------------------------
iter6_common_args='
--prev_explainer_chk_pt_folder ./checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1 ./checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2 ./checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter3 ./checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter4 ./checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter5
--checkpoint-residual model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar
--iter 6
--cov 0.2 0.2 0.2 0.2 0.2 0.2
--lr 0.01 0.01 0.01 0.01 0.01 0.01
'
python ./src/codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar"  "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" "model_g_best_model_epoch_345.pth.tar" \
    --expert-to-train "explainer" \
    $iter6_common_args \
    $common_args \
    > $slurm_output_iter6_g_train


python ./src/codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" "model_g_best_model_epoch_345.pth.tar" "model_g_best_model_epoch_87.pth.tar" \
    --expert-to-train "explainer" \
    $iter6_common_args \
    $common_args \
    > $slurm_output_iter6_g_test


python ./src/codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" "model_g_best_model_epoch_345.pth.tar" "model_g_best_model_epoch_87.pth.tar" \
    --expert-to-train "residual" \
    $iter6_common_args \
    $common_args \
    > $slurm_output_iter5_residual_train


# Train final residual
python ./src/codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" "model_g_best_model_epoch_345.pth.tar" "model_g_best_model_epoch_87.pth.tar" \
    --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
    --expert-to-train "residual" \
    $iter6_common_args \
    $common_args \
    > $slurm_output_iter6_residual_train

# ---------------------------------
# Explanations
# ---------------------------------
# Update ./src/codebase/Completeness_and_interventions/paths_MoIE.json file with appropriate paths for the checkpoints and outputs
python ./src/codebase/FOLs_vision_main.py --arch "ResNet101" --dataset "cub" --iterations 6  > $slurm_explanations
