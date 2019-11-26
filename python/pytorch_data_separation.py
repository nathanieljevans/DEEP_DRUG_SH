'''
This is the last step to generating the data we will use for the deep learning project.

This script:
  1. separates expression data into individual files
    a.
  2. transforms gene targets into vector encoding
'''

import pandas as pd
import torch
import os
import zipfile
import numpy as np
import pickle
import time

class expr_data:
    def __init__(self, depmap_path, beataml_path, gene_order):
        '''
        load expr data into memory
        '''
        self.order = gene_order
        self.depmap =  pd.read_csv(depmap_path)
        self.beataml = pd.read_csv(beataml_path)
        self.holder = dict()

    def get_expr(self, id, source, gene_order):
        '''
        source <str> [beataml, depmap] - which expression dataset to retrieve from
        genes <list> - entrez gene ids, in the order to have expression values listed
        '''
        if id in self.holder:
            print(f'returning from memory, size of holder: {len(self.holder)}')
            return self.holder[id]
        else:
            print('returning novel expr')
            if source == 'DepMap_ID':
                df = self.depmap[self.depmap.cell_line == id]
                arr = np.array([self.handle_me_this(df[df.ensembl_id == g].expression) for g in self.order])
                self.holder[id] = arr
                return arr
            elif source == 'lab_id':
                df = self.depmap[self.beataml.lab_id == id]
                arr = np.array([self.handle_me_this(df[df.ensembl_id == g].expression) for g in self.order])
                self.holder[id] = arr
                return arr
            else:
                raise ValueError(f'source must be [DepMap_ID, lab_id], got: {source}')

    def handle_me_this(self, expr):
        '''

        '''
        if len(expr > 0):
            expr = expr.values[0]
        else:
            expr = 0
        return expr


class drug_data:
    def __init__(self, path, toensembl):
        '''
        load drug data into memory
        '''
        self.drug = pd.read_csv(path)
        self.toensembl = toensembl

    def get_targets(self, gene_order):
        '''
        gene_order <list> - ensembl gene ids, in the order to have target (boolean) listed

        make this a generator: yield (targets, id, data source, response type, response)
        '''
        for _, tHGNC, tENTREZ, drugname, response, response_type, id, id_type, tENSEMBL in self.drug.values:
            if (id_type == 'DepMap_ID'):
                targets = [1*(tENSEMBL==g) for g in gene_order]
            else:
                ## beatAML targets (may be multiple)
                targets = tHGNC
                if ';' in targets:
                    targets = targets.split(';')
                else:
                    targets = [targets]
                targets = [1 if self.toensembl[g] in targets else 0 for g in gene_order]

            yield (targets, id, id_type, response_type, response)


class file_name_tracker:

    def __init__ (self, start=0):
        '''
        start <int> - unique identifier; for use if appending to current data sets
        '''
        self.incr = start
        self.label_dict = dict()

    def get_name(self, id, source, response_type, response):
        '''
        source <str> - dataset [beataml, gdsc, ccle, ctrp, etc]
        response_type <str> - type of reponse variable [gene_dependency, AUC, drug_fold_change]
        response <float>

        return the the file name to use for the given observation; save the name to dictionary, linking the response

        '''
        self.label_dict[self.incr] = (id, source, response_type, response)
        self.incr += 1
        return self.incr - 1

    def get_label_dict(self):
        '''
        return a dictionary linking file names -> response values
        '''
        return self.label_dict

def unpack_data2(p='../data2.zip', o='./../'):
    '''
    '''
    if not os.path.exists(p[:-4]):
        print('unpacking data2.zip...')
        with zipfile.ZipFile(p, 'r') as zip_ref:
            zip_ref.extractall(o)
    else:
        print('data2 already unpacked.')

def make_toensembl(p):
    '''

    '''
    idmap = pd.read_csv(p)
    print(idmap.head())
    toensembl = dict()
    for _,entrez,HGNC,ensembl in idmap.values:
        #print(gene)
        toensembl[HGNC] = ensembl

    return toensembl

if __name__ == '__main__':
    '''

    '''
    unpack_data2()

    order = pd.read_csv('../data2/aml_genes_to_use.csv')['x'].values.tolist() ## load gene order (genes_to_use)
    print(order[0:5])
    ngenes = len(order)
    print(ngenes)
    obs_size = (ngenes, 2)

    toensembl = make_toensembl('../data2/gene_id_map.csv')

    expr = expr_data(depmap_path = '../data2/depmap_expr_amlgenes.csv', beataml_path = '../data2/beataml_expr_amlgenes.csv', gene_order = order)
    drug = drug_data('../data2/drug_data_aml_genes.csv', toensembl)
    namer = file_name_tracker()

    ii = 0
    for targets, id, id_type, response_type, response in drug.get_targets(order):
        print(f'working ... count={ii}', end='\r')
        obs = np.zeros(obs_size)
        obs[:,0] = expr.get_expr(id=id, source=id_type, gene_order=order)
        obs[:,1] = targets
        obs_name = namer.get_name(id, id_type, response_type, response)
        pyt = torch.Tensor(obs)
        torch.save(pyt, f'../data_pytorch/tensors/{obs_name}.pt')
        ii += 1
        if ii == 5: break

    label_dict = namer.get_label_dict()
    f = open("../data_pytorch/label_dict.pkl","wb")
    pickle.dump(label_dict,f)
    f.close()

    print(label_dict)
