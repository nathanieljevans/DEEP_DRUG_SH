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

'H1' : 1000,
'H2' : 500,
'H3' : 100,                # Layer 3 - Dataset Shared
'DH' : 100,                # Dataset Specific Layer

'DO' : 0.,                # Dropout
'NCONVS' : 2,
'PRINT_EVERY' : 1,
'NGENES' : 523, #/ 523
'EPOCHS' : 10,
'LEARNING_WEIGHT' : 5e-2,
'WEIGHT_DECAY' : 0.001,
'LR_DECAY_PATIENCE' : 3,
'PRETRAIN_EPOCHS' : 3,
'PRETRAIN_LR' : 1,
'PRETRAIN_WD' : 0.01,
'PRETRAIN_DO' : 0.1,
'PRETRAIN_MSE_WEIGHT' : 1000,     # Weight applied to target MSE
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
