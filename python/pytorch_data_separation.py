'''
This is the last step to generating the data we will use for the deep learning project.

This script:
  1. separates expression data into individual files
    a.
  2. transforms gene targets into vector encoding
'''

import pandas as pd
import torch


class expr_data:
    def __init__(self, path):
        '''
        load expr data into memory
        '''
        pass

    def get_expr(source, gene_order):
        '''
        source <str> [beataml, depmap] - which expression dataset to retrieve from
        genes <list> - entrez gene ids, in the order to have expression values listed
        '''
        pass

class drug_data:
    def __init__(self, path):
        '''
        load drug data into memory
        '''
        pass

    def get_targets(genes, gene_order):
        '''
        genes <list> - entrez target genes
        gene_order <list> - entrez gene ids, in the order to have target (boolean) listed

        make this a generator: yield (targets, data source, response type, response)
        '''
        pass

class file_name_tracker:

    def __init__ (self, start=0):
        '''
        start <int> - unique identifier; for use if appending to current data sets
        '''
        self.start = start

    def get_name(self, source, response_type, response):
        '''
        source <str> - dataset [beataml, gdsc, ccle, ctrp, etc]
        response_type <str> - type of reponse variable [gene_dependency, AUC, drug_fold_change]
        response <float>

        return the the file name to use for the given observation; save the name to dictionary, linking the response

        name = <source>_<response_type>_<id_incrementor>
        '''
        pass

    def get_label_dict(self):
        '''
        return a dictionary linking file names -> response values
        '''
        pass


if __name__ == '__main__':
    '''

    '''
    ngenes = 507
    obs_size = (ngenes, 2)

    drug_data_path = ''
    expr_data_path = ''

    order = ## load gene order (genes_to_use)

    drug = drug_data(drug_data_path)
    expr = expr_data(expr_data_path)
    namer = file_name_tracker()

    for targets, source, response_type, response in drug.get_targets():
        obs = np.zeros(obs_size)
        obs[:,0] = expr(source, order)
        obs[:,1] = targets
        name = namer.get_name(source, response_type, response)
        pyt = torch.Tensor(obs)
        torch.save_tensor(pyt)

    label_dict = namer.get_label_dict() 
