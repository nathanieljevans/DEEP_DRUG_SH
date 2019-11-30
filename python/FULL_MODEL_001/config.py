'''

'''
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
DATA_DIR = '../../data_pytorch/tensors/'
LABEL_PATH = '../../data_pytorch/label_dict.pkl'
SPLIT_LABEL_PATH = '../../data_pytorch/split_label_dict.pkl'
MODEL_OUT_DIR = '../../models/'
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
RESPLIT_DATA = False
TRAIN_PROP, TEST_PROP, VAL_PROP = (0.6, 0.2, 0.2)
N_BEATAML_PATIENTS_EXCLUSIVE_TO_TEST = 20            # remove patients for test
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    ## `i` indicates the order of classification layers
RESP_TYPES = {x:i for i,x in enumerate(['RNAi_dependency',
                                        'crispr_dependency',
                                        'pooled_drugresp_prism',
                                        'AUC_GDSC','CTRP_AUC',
                                        'AUC_drug_CCLE',
                                        'beatAML_AUC'])}
N_DATATYPES = len(RESP_TYPES)

H1 = 1000
H2 = 500
H3 = 250
DH = 100                # Dataset Specific Layer

DO = 0.1                # Dropout
NCONVS = 2
PRINT_EVERY = 1
NGENES = 523 #/ 523
EPOCHS = 5
LEARNING_WEIGHT = 1e-2
WEIGHT_DECAY = 0.05
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
REPRODUCIBLE = True
SEED = 0



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
train_params = {'batch_size': 2048,
          'shuffle': True,
          'num_workers': 3}

test_params = {'batch_size': 512,
          'shuffle': False,
          'num_workers': 0}
