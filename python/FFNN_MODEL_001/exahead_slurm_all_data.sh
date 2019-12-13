#!/bin/bash

#SBATCH --job-name=all09
#SBATCH --time=36:00:00
#SBATCH --mem=64G
#SBATCH -c 20
#SBATCH --out=./../../data_pytorch/logfile/cpu_alldata_train08_LOG.txt
#SBATCH --error=./../../data_pytorch/logfile/cpu_alldata_train08_ERROR.txt

python ~/DEEP_DRUG_SH/python/FFNN_MODEL_001/train.py --config config--all_data
