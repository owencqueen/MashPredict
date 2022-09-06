#!/bin/bash
#SBATCH -A ISAAC-UTK0196
#SBATCH --partition=campus
#SBATCH --nodes=1 
#SBATCH --qos=campus
#SBATCH --ntasks-per-node=4
#SBATCH --time=04:00:00
#SBATCH -o /lustre/isaac/scratch/oqueen/MashPredict/nn_models/cluster_runs/outputs/s500/fold=CVNUM.o
#SBATCH -e /lustre/isaac/scratch/oqueen/MashPredict/nn_models/cluster_runs/outputs/s500/fold=CVNUM.e
#SBATCH -J s500_CVNUM

base="/lustre/isaac/scratch/oqueen/MashPredict/nn_models"

cd $base
conda activate /lustre/isaac/scratch/oqueen/codonbert

python3 mashnet_cv.py --cv CVNUM --mashsize 500