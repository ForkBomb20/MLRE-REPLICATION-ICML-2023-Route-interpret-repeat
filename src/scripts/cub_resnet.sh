#!/bin/sh
#SBATCH --output=path/cub_resnet_%j.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT

slurm_output_bb_train=cub_resnet_bb_train_$CURRENT.out
slurm_output_bb_test=cub_resnet_bb_test_$CURRENT.out
slurm_output_t_train=cub_resnet_t_train_$CURRENT.out
slurm_output_t_test=cub_resnet_t_test_$CURRENT.out

slurm_output_iter1_g_train=cub_resnet_iter1_g_train_$CURRENT.out
slurm_output_iter1_g_test=cub_resnet_iter1_g_test_$CURRENT.out
slurm_output_iter1_residual_train=cub_resnet_iter1_residual_train_$CURRENT.out

slurm_output_iter2_g_train=cub_resnet_iter2_g_train_$CURRENT.out
slurm_output_iter2_g_test=cub_resnet_iter2_g_test_$CURRENT.out
slurm_output_iter2_residual_train=cub_resnet_iter2_residual_train_$CURRENT.out

slurm_output_iter3_g_train=cub_resnet_iter3_g_train_$CURRENT.out
slurm_output_iter3_g_test=cub_resnet_iter3_g_test_$CURRENT.out
slurm_output_iter3_residual_train=cub_resnet_iter3_residual_train_$CURRENT.out

slurm_output_iter4_g_train=cub_resnet_iter4_g_train_$CURRENT.out
slurm_output_iter4_g_test=cub_resnet_iter4_g_test_$CURRENT.out
slurm_output_iter4_residual_train=cub_resnet_iter4_residual_train_$CURRENT.out

slurm_output_iter5_g_train=cub_resnet_iter5_g_train_$CURRENT.out
slurm_output_iter5_g_test=cub_resnet_iter5_g_test_$CURRENT.out
slurm_output_iter5_residual_train=cub_resnet_iter5_residual_train_$CURRENT.out

slurm_output_iter6_g_train=cub_resnet_iter6_g_train_$CURRENT.out
slurm_output_iter6_g_test=cub_resnet_iter6_g_test_$CURRENT.out
slurm_output_iter6_residual_train=cub_resnet_iter6_residual_train_$CURRENT.out
slurm_output_iter6_residual_test=cub_resnet_iter6_residual_test_$CURRENT.out

slurm_explanations=cub_resnet_explanations_$CURRENT.out

echo "CUB-200 ResNet101"
source path-of-conda/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7_rtx_6000

# BB model
# BB Training scripts

python ../codebase/train_BB_CUB.py \
    --bs 16 \
    --arch "ResNet101" \
    > $slurm_output_bb_train


# BB Testing scripts
# Update ../codebase/Completeness_and_interventions/paths_MoIE.json file with appropriate paths for the checkpoints and outputs
python ../codebase/test_BB_CUB.py \
    --checkpoint-file "best_model_epoch_63.pth.tar" \
    --save-activations True \
    --layer "layer4" \
    --bs 16 \
    --arch "ResNet101" \
    > $slurm_output_bb_test


# T model
# train
python ../codebase/train_t_CUB.py \
    --checkpoint-file "best_model_epoch_63.pth.tar" \
    --bs 32 \
    --layer "layer4" \
    --flattening-type "adaptive" \
    --arch "ResNet101" \
    > $slurm_output_t_train

# Test
python ../codebase/test_t_CUB.py \
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
python ../codebase/train_explainer_CUB.py \
    --expert-to-train "explainer" \
    $iter1_common_args \
    $common_args \
    > $slurm_output_iter1_g_train

# Test explainer
python ../codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" \
    --expert-to-train "explainer" \
    $iter1_common_args \
    $common_args \
    >  $slurm_output_iter1_g_test

# Train residual
python ../codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" \
    --expert-to-train "residual" \
    $iter1_common_args \
    $common_args \
    > $slurm_output_iter1_residual_train

#---------------------------------
# iter 2
#---------------------------------
iter2_common_args='
--prev_explainer_chk_pt_folder /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1
--checkpoint-residual model_residual_best_model_epoch_1.pth.tar
--iter 2
--cov 0.2 0.2
--lr 0.01 0.01
'

python ../codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" \
    --expert-to-train "explainer" \
    $iter2_common_args \
    $common_args \
    >  $slurm_output_iter2_g_train

python ../codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" \
    --expert-to-train "explainer" \
    $iter2_common_args \
    $common_args \
    > $slurm_output_iter2_g_test

python ../codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" \
    --expert-to-train "residual" \
    $iter2_common_args \
    $common_args \
    > $slurm_output_iter2_residual_train

#---------------------------------
# iter 3
#---------------------------------
iter3_common_args='
--prev_explainer_chk_pt_folder /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1 /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2
--checkpoint-residual model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar
--iter 3
--cov 0.2 0.2 0.2
--lr 0.01 0.01 0.01
'

python ../codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar"  "model_g_best_model_epoch_188.pth.tar" \
    --expert-to-train "explainer" \
    $iter3_common_args \
    $common_args \
    > $slurm_output_iter3_g_train


python ../codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" \
    --expert-to-train "explainer" \
    $iter3_common_args \
    $common_args \
    > $slurm_output_iter3_g_test


python ../codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" \
    --expert-to-train "residual" \
    $iter3_common_args \
    $common_args \
    > $slurm_output_iter3_residual_train

#---------------------------------
# iter 4
#---------------------------------
iter4_common_args='
--prev_explainer_chk_pt_folder /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1 /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2 /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter3
--checkpoint-residual model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar
--iter 4
--cov 0.2 0.2 0.2 0.2
--lr 0.01 0.01 0.01 0.01
'
python ../codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" \
    --expert-to-train "explainer" \
    $iter4_common_args \
    $common_args \
    > $slurm_output_iter4_g_train


python ../codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" \
    --expert-to-train "explainer" \
    $iter4_common_args \
    $common_args \
    > $slurm_output_iter4_g_test


python ../codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" \
    --expert-to-train "residual" \
    $iter4_common_args \
    $common_args \
    > $slurm_output_iter4_residual_train

# ---------------------------------
# iter 5
# ---------------------------------
iter5_common_args='
--prev_explainer_chk_pt_folder /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1 /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2 /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter3 /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter4
--checkpoint-residual model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar
--iter 5
--cov 0.2 0.2 0.2 0.2 0.2
--lr 0.01 0.01 0.01 0.01 0.01
'
python ../codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar"  "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" \
    --expert-to-train "explainer" \
    $iter5_common_args \
    $common_args \
    > $slurm_output_iter5_g_train


python ../codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" "model_g_best_model_epoch_345.pth.tar" \
    --expert-to-train "explainer" \
    $iter5_common_args \
    $common_args \
    > $slurm_output_iter5_g_test


python ../codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" "model_g_best_model_epoch_345.pth.tar" \
    --expert-to-train "residual" \
    $iter5_common_args \
    $common_args \
    > $slurm_output_iter5_residual_train

# ---------------------------------
# iter 6
# ---------------------------------
iter6_common_args='
--prev_explainer_chk_pt_folder /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/iter1 /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter2 /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter3 /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter4 /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/cub/explainer/ResNet101/lr_0.01_epochs_500_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01/iter5
--checkpoint-residual model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar model_residual_best_model_epoch_1.pth.tar
--iter 6
--cov 0.2 0.2 0.2 0.2 0.2 0.2
--lr 0.01 0.01 0.01 0.01 0.01 0.01
'
python ../codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar"  "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" "model_g_best_model_epoch_345.pth.tar" \
    --expert-to-train "explainer" \
    $iter6_common_args \
    $common_args \
    > $slurm_output_iter6_g_train


python ../codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" "model_g_best_model_epoch_345.pth.tar" "model_g_best_model_epoch_87.pth.tar" \
    --expert-to-train "explainer" \
    $iter6_common_args \
    $common_args \
    > $slurm_output_iter6_g_test


python ../codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" "model_g_best_model_epoch_345.pth.tar" "model_g_best_model_epoch_87.pth.tar" \
    --expert-to-train "residual" \
    $iter6_common_args \
    $common_args \
    > $slurm_output_iter5_residual_train


# Train final residual
python ../codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model_epoch_64.pth.tar" "model_g_best_model_epoch_188.pth.tar" "model_g_best_model_epoch_110.pth.tar" "model_g_best_model_epoch_257.pth.tar" "model_g_best_model_epoch_345.pth.tar" "model_g_best_model_epoch_87.pth.tar" \
    --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
    --expert-to-train "residual" \
    $iter6_common_args \
    $common_args \
    > $slurm_output_iter6_residual_train

# ---------------------------------
# Explanations
# ---------------------------------
# Update ../codebase/Completeness_and_interventions/paths_MoIE.json file with appropriate paths for the checkpoints and outputs
python ../codebase/FOLs_vision_main.py --arch "ResNet101" --dataset "cub" --iterations 6  > $slurm_explanations
