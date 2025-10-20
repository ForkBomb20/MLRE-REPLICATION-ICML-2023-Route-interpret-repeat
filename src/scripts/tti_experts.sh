#!/bin/sh
#SBATCH --output=path/test_time_interventions_difficult_expert_%j.out
pwd; hostname; date
CURRENT=`date +"%Y-%m-%d_%T"`
echo $CURRENT

slurm_output_cub_resnet=cub_resnet_$CURRENT.out

echo "Test Time Interventions for difficult samples covered by later experts"
source path-of-conda/anaconda3/etc/profile.d/conda.sh
conda activate python_3_7_rtx_6000

python ../codebase/test_time_interventions_main.py --expert_driven_interventions "y" --expert 5 6 --model "MoIE" --arch "ResNet101" --dataset "cub" --iterations 6 --top_K 3 5 10 15 20 25 30 50 75 90 108 > $slurm_output_cub_resnet
