#!/bin/bash
#SBATCH -N 1 # number of minimum nodes
#SBATCH -c 2 # number of cores
#SBATCH --gres=gpu:1 # Request 1 gpu
#SBATCH --mail-user=itay.zach@campus.technion.ac.il
# (change to your own email if you wish to get one, or just delete this and the following lines)
#SBATCH --mail-type=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job-name="hw1"
#SBATCH -o out/sbatch.%j.out # stdout goes here
#SBATCH -e out/sbatch.%j.out # stderr goes here
echo "running $@"
python main.py $@
