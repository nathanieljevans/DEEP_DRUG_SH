#!/bin/bash

#SBATCH --job-name=aml02
#SBATCH --time=06:00:00
#SBATCH --mem=12G
#SBATCH -c 12
#SBATCH --out=./../../data_pytorch/logfile/cpu_beataml_train_LOG002.txt
#SBATCH --error=./../../data_pytorch/logfile/cpu_beataml_train_ERROR002.txt

python ~/DEEP_DRUG_SH/python/FFNN_MODEL_001/train.py --config config--beatAML_data

