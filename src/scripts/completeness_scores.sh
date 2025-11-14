#!/bin/sh
#SBATCH --output=path/completeness_scores_%j.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT

slurm_output_cub_resnet_concept_mask=cub_resnet_concept_mask_$CURRENT.out
slurm_output_cub_resnet_concept_completeness=cub_resnet_concept_completeness_$CURRENT.out

echo "Completeness Scores"
module load python/3.13.2
echo "CUB-200 ResNet101"
source ./env/bin/activate
which python

# MoIE

python ./src/codebase/concept_completeness_concept_mask_main.py --model "MoIE" --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 108 > $slurm_output_cub_resnet_concept_mask

python ./src/codebase/concept_completeness_main.py --model "MoIE" --epochs 75 --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 108 > $slurm_output_cub_resnet_concept_completeness
