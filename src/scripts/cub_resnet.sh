#!/bin/sh
#SBATCH --job-name=cub_resnet
#SBATCH --output=slurm_outs/cub_resnet_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --account=eecs498f25s007_class

pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT

# Output paths
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

# Environment setup
module load python/3.13.2
echo "CUB-200 ResNet101"
source ./env/bin/activate
which python

# Dataset check
if [ ! -d /tmp/$USER/data ]; then
    echo "Copying dataset to local /tmp..."
    mkdir -p /tmp/$USER/data
    rsync -ah --info=progress2 --ignore-existing ./data/ /tmp/$USER/data
    echo "Dataset copied to local /tmp."
else
    echo "Using cached local dataset."
fi

# ------------------------------------------------
# BB model
# ------------------------------------------------
echo "[INFO] Would run: python ./src/codebase/train_BB_CUB.py ..."
# python ./src/codebase/train_BB_CUB.py \
#     --bs 16 \
#     --arch "ResNet101" \
#     --data-root "/tmp/$USER/data/CUB_200_2011" \
#     > $slurm_output_bb_train

echo "[INFO] Would run: python ./src/codebase/test_BB_CUB.py ..."
# python ./src/codebase/test_BB_CUB.py \
#     --checkpoint-file "best_model.pth.tar" \
#     --save-activations True \
#     --layer "layer4" \
#     --bs 16 \
#     --arch "ResNet101" \
#     --data-root "/tmp/$USER/data/CUB_200_2011" \
#     > $slurm_output_bb_test

# ------------------------------------------------
# T model
# ------------------------------------------------
echo "[INFO] Would run: python ./src/codebase/train_t_CUB.py ..."
# python ./src/codebase/train_t_CUB.py \
#     --checkpoint-file "best_model.pth.tar" \
#     --bs 32 \
#     --layer "layer4" \
#     --flattening-type "adaptive" \
#     --arch "ResNet101" \
#     --data-root "/tmp/$USER/data/CUB_200_2011" \
#     > $slurm_output_t_train

echo "[INFO] Would run: python ./src/codebase/test_t_CUB.py ..."
# python ./src/codebase/test_t_CUB.py \
#     --checkpoint-file "best_model.pth.tar" \
#     --checkpoint-file-t "best_model.pth.tar" \
#     --save-concepts True \
#     --bs 16 \
#     --solver-LR "sgd" \
#     --loss-LR "BCE" \
#     --layer "layer4" \
#     --flattening-type "adaptive" \
#     --arch "ResNet101" \
#     --data-root "/tmp/$USER/data/CUB_200_2011" \
#     > $slurm_output_t_test

# ------------------------------------------------
# MoIE Training
# ------------------------------------------------
common_args='
--root-bb lr_0.001_epochs_95
--checkpoint-bb best_model.pth.tar
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
--data-root /tmp/$USER/data/CUB_200_2011
'

#---------------------------------
# Iteration 1
#---------------------------------
iter1_common_args='
--iter 1
--cov 0.2
--lr 0.01
'

echo "[RUNNING] Iter 1 Explainer Train"
python ./src/codebase/train_explainer_CUB.py \
    --expert-to-train "explainer" \
    $iter1_common_args \
    $common_args \
    --epochs 120 \
    > $slurm_output_iter1_g_train

echo "[RUNNING] Iter 1 Explainer Test"
python ./src/codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model.pth.tar" \
    --expert-to-train "explainer" \
    $iter1_common_args \
    $common_args \
    > $slurm_output_iter1_g_test

echo "[RUNNING] Iter 1 Residual Train"
python ./src/codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model.pth.tar" \
    --expert-to-train "residual" \
    $iter1_common_args \
    $common_args \
    > $slurm_output_iter1_residual_train

#---------------------------------
# Iteration 2
#---------------------------------
iter2_common_args='
--iter 2
--cov 0.4
--lr 0.01
'

echo "[RUNNING] Iter 2 Explainer Train"
python ./src/codebase/train_explainer_CUB.py \
    --expert-to-train "explainer" \
    $iter2_common_args \
    $common_args \
    --epochs 120 \
    > $slurm_output_iter2_g_train

echo "[RUNNING] Iter 2 Explainer Test"
python ./src/codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model.pth.tar" \
    --expert-to-train "explainer" \
    $iter2_common_args \
    $common_args \
    > $slurm_output_iter2_g_test

echo "[RUNNING] Iter 2 Residual Train"
python ./src/codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model.pth.tar" \
    --expert-to-train "residual" \
    $iter2_common_args \
    $common_args \
    > $slurm_output_iter2_residual_train

#---------------------------------
# Iteration 3
#---------------------------------
iter3_common_args='
--iter 3
--cov 0.6
--lr 0.01
'

echo "[RUNNING] Iter 3 Explainer Train"
python ./src/codebase/train_explainer_CUB.py \
    --expert-to-train "explainer" \
    $iter3_common_args \
    $common_args \
    --epochs 120 \
    > $slurm_output_iter3_g_train

echo "[RUNNING] Iter 3 Explainer Test"
python ./src/codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model.pth.tar" \
    --expert-to-train "explainer" \
    $iter3_common_args \
    $common_args \
    > $slurm_output_iter3_g_test

echo "[RUNNING] Iter 3 Residual Train"
python ./src/codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model.pth.tar" \
    --expert-to-train "residual" \
    $iter3_common_args \
    $common_args \
    > $slurm_output_iter3_residual_train

#---------------------------------
# Iteration 4
#---------------------------------
iter4_common_args='
--iter 4
--cov 0.7
--lr 0.01
'

echo "[RUNNING] Iter 4 Explainer Train"
python ./src/codebase/train_explainer_CUB.py \
    --expert-to-train "explainer" \
    $iter4_common_args \
    $common_args \
    --epochs 120 \
    > $slurm_output_iter4_g_train

echo "[RUNNING] Iter 4 Explainer Test"
python ./src/codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model.pth.tar" \
    --expert-to-train "explainer" \
    $iter4_common_args \
    $common_args \
    > $slurm_output_iter4_g_test

echo "[RUNNING] Iter 4 Residual Train"
python ./src/codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model.pth.tar" \
    --expert-to-train "residual" \
    $iter4_common_args \
    $common_args \
    > $slurm_output_iter4_residual_train

#---------------------------------
# Iteration 5
#---------------------------------
iter5_common_args='
--iter 5
--cov 0.8
--lr 0.01
'

echo "[RUNNING] Iter 5 Explainer Train"
python ./src/codebase/train_explainer_CUB.py \
    --expert-to-train "explainer" \
    $iter5_common_args \
    $common_args \
    --epochs 120 \
    > $slurm_output_iter5_g_train

echo "[RUNNING] Iter 5 Explainer Test"
python ./src/codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model.pth.tar" \
    --expert-to-train "explainer" \
    $iter5_common_args \
    $common_args \
    > $slurm_output_iter5_g_test

echo "[RUNNING] Iter 5 Residual Train"
python ./src/codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model.pth.tar" \
    --expert-to-train "residual" \
    $iter5_common_args \
    $common_args \
    > $slurm_output_iter5_residual_train

#---------------------------------
# Iteration 6
#---------------------------------
iter6_common_args='
--iter 6
--cov 0.9
--lr 0.01
'

echo "[RUNNING] Iter 6 Explainer Train"
python ./src/codebase/train_explainer_CUB.py \
    --expert-to-train "explainer" \
    $iter6_common_args \
    $common_args \
    --epochs 120 \
    > $slurm_output_iter6_g_train

echo "[RUNNING] Iter 6 Explainer Test"
python ./src/codebase/test_explainer_CUB.py \
    --checkpoint-model "model_g_best_model.pth.tar" \
    --expert-to-train "explainer" \
    $iter6_common_args \
    $common_args \
    > $slurm_output_iter6_g_test

echo "[RUNNING] Iter 6 Residual Train"
python ./src/codebase/train_explainer_CUB.py \
    --checkpoint-model "model_g_best_model.pth.tar" \
    --expert-to-train "residual" \
    $iter6_common_args \
    $common_args \
    > $slurm_output_iter6_residual_train

#---------------------------------
# Final explanations
#---------------------------------
echo "[RUNNING] Final Explanation Generation"
python ./src/codebase/FOLs_vision_main.py \
    --arch "ResNet101" \
    --dataset "cub" \
    --iterations 6 \
    > $slurm_explanations
