'''

'''
params = {
'NAME' : 'ALL-DATA-3FC-1O',

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


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    ## `i` indicates the order of classification layers
'RESP_TYPES' : {x:i for i,x in enumerate(['RNAi_dependency',
                                        'crispr_dependency',
                                        'pooled_drugresp_prism',
                                        'AUC_GDSC','CTRP_AUC',
                                        'AUC_drug_CCLE',
                                        'beatAML_AUC'])},
'N_DATATYPES' : 7,

'H1' : 2000,
'H2' : 1000,
'H3' : 100,                # Layer 3 - Dataset Shared
'DH' : 500,                # Dataset Specific Layer

'DO' : 0.25,                # Dropout
'NCONVS' : 10,
'PRINT_EVERY' : 1,
'NGENES' : 523, #/ 523
'EPOCHS' : 2,
'LEARNING_WEIGHT' : 1e-1,
'WEIGHT_DECAY' : 0.001,
'LR_DECAY_PATIENCE' : 50,       # batches (not epochs)
'PRETRAIN_EPOCHS' : 1,
'PRETRAIN_LR' : 1e-1,
'PRETRAIN_WD' : 0.01,
'PRETRAIN_DO' : 0.9,
'PRETRAIN_MSE_WEIGHT' : 10,     # Weight applied to target MSE
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
'REPRODUCIBLE' : True,
'SEED' : 1000,

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
'train_params' : {'batch_size': 4*2048,
          'shuffle': True,
          'num_workers': 6},

'test_params' : {'batch_size': 2048,
          'shuffle': False,
          'num_workers': 6}
}
