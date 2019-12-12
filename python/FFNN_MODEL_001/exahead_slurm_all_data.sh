#!/bin/bash

#SBATCH --job-name=all06
#SBATCH --time=36:00:00
#SBATCH --mem=16G
#SBATCH -c 12
#SBATCH --out=./../../data_pytorch/logfile/cpu_alldata_train_LOG.txt
#SBATCH --error=./../../data_pytorch/logfile/cpu_alldata_train_ERROR.txt

python ~/DEEP_DRUG_SH/python/FFNN_MODEL_001/train.py --config config--all_data
