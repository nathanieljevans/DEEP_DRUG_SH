#!/bin/bash

#SBATCH --job-name=aml_01
#SBATCH --time=06:00:00
#SBATCH --mem=12G
#SBATCH -c 12
#SBATCH --out=./../../data_pytorch/logfile/cpu_beataml_train_LOG001.txt
#SBATCH --error=./../../data_pytorch/logfile/cpu_beataml_train_ERROR001.txt

python ~/DEEP_DRUG_SH/python/FFNN_MODEL_001/train.py --config config--beatAML_data

