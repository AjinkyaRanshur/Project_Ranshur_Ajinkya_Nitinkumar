#!/bin/bash

#SBATCH --job-name=ar_vgg16_model
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --error=ar_vgg16_model.err
#SBATCH --output=ar_vgg16_model.out
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=ranshur.ajinkya@students.iiserpune.ac.in
#SBATCH --mail-type=END

cd $SLURM_SUBMIT_DIR

conda info

old_ifs=$IFS
IFS='
'
for cmd in `cat cmd_list`
do
srun -N1 -n1 -c1 --exclusive bash -c ${cmd} &
done
wait
IFS=$old_ifs

