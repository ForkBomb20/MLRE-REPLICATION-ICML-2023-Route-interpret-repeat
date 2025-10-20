#!/bin/sh
#SBATCH --output=path/completeness_scores_%j.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT

slurm_output_cub_vit_concept_mask=cub_vit_concept_mask_$CURRENT.out
slurm_output_cub_vit_concept_completeness=cub_vit_concept_completeness_$CURRENT.out
slurm_output_cub_resnet_concept_mask=cub_resnet_concept_mask_$CURRENT.out
slurm_output_cub_resnet_concept_completeness=cub_resnet_concept_completeness_$CURRENT.out
slurm_output_ham_concept_mask=ham_concept_mask_$CURRENT.out
slurm_output_ham_concept_completeness=ham_concept_completeness_$CURRENT.out
slurm_output_awa2_concept_mask=awa2_concept_mask_$CURRENT.out
slurm_output_awa2_concept_completeness=awa2_concept_completeness_$CURRENT.out

echo "Completeness Scores"
source path-of-conda/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7_rtx_6000

# -----------------------------------------------------
# CUB_ResNet101
# -----------------------------------------------------
# MoIE

python ../codebase/concept_completeness_concept_mask_main.py --model "MoIE" --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 108 > $slurm_output_cub_resnet_concept_mask

python ../codebase/concept_completeness_main.py --model "MoIE" --epochs 75 --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 108 > $slurm_output_cub_resnet_concept_completeness
