#!/bin/bash
#SBATCH --job-name=cpuTrainCNN
#SBATCH --time=12:00:00
#SBATCH --mem=25G
#SBATCH -c 8
#SBATCH --out=logfiles/cpuTrainCNN_out.txt
#SBATCH --error=logfiles/cpuTrainCNN_err.txt
python /home/groups/brainmri/ncanda/gareth/dl_3d_cnn/testReal/runTERF.py
