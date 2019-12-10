#!/bin/bash

#SBATCH --job-name=evans_cpu_alldata_train
#SBATCH --time=10:00:00
#SBATCH --mem=25G
#SBATCH -c 12
#SBATCH --out=./../../data_pytorch/logfile/cpu_alldata_train_LOG.txt
#SBATCH --error=./../../data_pytorch/logfile/cpu_alldata_train_ERROR.txt

python ~/DEEP_DRUG_SH/python/FFNN_MODEL_001/train.py --config config--all_data

