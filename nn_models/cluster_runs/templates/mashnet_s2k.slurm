#!/bin/bash
#SBATCH -A ISAAC-UTK0196
#SBATCH --partition=campus
##SBATCH --qos=campus-gpu
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
##SBATCH --gres=gpu:1
#SBATCH --qos=campus
#SBATCH --time=16:00:00
#SBATCH -o /lustre/isaac/scratch/oqueen/MashPredict/nn_models/cluster_runs/outputs/s2k/fold=CVNUM.o
#SBATCH -e /lustre/isaac/scratch/oqueen/MashPredict/nn_models/cluster_runs/outputs/s2k/fold=CVNUM.e
#SBATCH -J s2k_CVNUM

base="/lustre/isaac/scratch/oqueen/MashPredict/nn_models"

cd $base
conda activate /lustre/isaac/scratch/oqueen/codonbert

python3 mashnet_cv.py --cv CVNUM --mashsize 2k