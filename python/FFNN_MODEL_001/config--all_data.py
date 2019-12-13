'''

'''
params = {
'NAME' : 'ALL-DATA-3FC_010',

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
'DATA_DIR' : '../../data_pytorch/tensors/',
'LABEL_PATH' : '../../data_pytorch/label_dict.pkl',
'SPLIT_LABEL_PATH' : '../../data_pytorch/split_label_dict.pkl',
'MODEL_OUT_DIR' : '../../models/',
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
'RESPLIT_DATA' : False,
'TRAIN_PROP' : 0.7,
'TEST_PROP' : 0.15,
'VAL_PROP' : 0.15,
'N_BEATAML_PATIENTS_EXCLUSIVE_TO_TEST' : 30  ,          # remove patients for test
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
'SAVE_MODEL_EVERY' : 5, # epochs
'PRINT_MID_EPOCH_INFO' : False,
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    ## `i` indicates the order of classification layers
'RESP_TYPES' : {x:i for i,x in enumerate(['RNAi_dependency',
                                        'crispr_dependency',
                                        'pooled_drugresp_prism',
                                        'AUC_GDSC','CTRP_AUC',
                                        'AUC_drug_CCLE',
                                        'beatAML_AUC'])},

'H1' : 5000,
'H2' : 5000,
'H3' : 50,                # Layer 3 - Dataset Shared
'DH' : 200,                # Dataset Specific Layer

'DO' : 0.5,                # Dropout
'NCONVS' : 10,
'PRINT_EVERY' : 1,
'NGENES' : 523, #/ 523
'EPOCHS' : 100,
'LEARNING_WEIGHT' : 1e-1,
'WEIGHT_DECAY' : 0.01,
'LR_DECAY_PATIENCE' : 20,       # batches (not epochs)

'PRETRAIN_EPOCHS' : 1,
'PRETRAIN_LR' : 1e-1,
'PRETRAIN_WD' : 0.1,
'PRETRAIN_DO' : 0.9,
'PRETRAIN_MSE_WEIGHT' : 50,     # Weight applied to target MSE
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
'REPRODUCIBLE' : False,
'SEED' : 1000,

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
'train_params' : {'batch_size': 2**14, #  16384
          'shuffle': True,
          'num_workers': 12},

'test_params' : {'batch_size': 2**14,
          'shuffle': False,
          'num_workers': 12}
}
