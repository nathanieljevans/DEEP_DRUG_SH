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
        self.failures = []

    def get_expr(self, id, source, gene_order):
        '''
        source <str> [beataml, depmap] - which expression dataset to retrieve from
        genes <list> - entrez gene ids, in the order to have expression values listed
        '''
        try:
            if id in self.holder:
                #print(f'returning from memory, size of holder: {len(self.holder)}')
                return self.holder[id]
            else:
                #print('returning novel expr')
                if source == 'DepMap_ID':
                    df = self.depmap[self.depmap.cell_line == id]
                    arr = np.array([self.handle_me_this(df[df.ensembl_id == g].expression) for g in self.order])
                    self.holder[id] = arr
                elif source == 'lab_id':
                    df = self.beataml[self.beataml.lab_id == id]
                    arr = np.array([self.handle_me_this(df[df.ensembl_id == g].expression) for g in self.order])
                    self.holder[id] = arr
                else:
                    raise ValueError(f'source must be [DepMap_ID, lab_id], got: {source}')

            assert np.sum(arr) > 0, 'expression vector is zero'
            return arr
        except AssertionError:
            self.failures.append(id)

    def handle_me_this(self, expr):
        '''

        '''
        if len(expr > 0):
            expr = expr.values[0]
        else:
            expr = 0
        return expr


class drug_data:
    def __init__(self, path, toensembl, start=0):
        '''
        load drug data into memory
        '''
        self.drug = pd.read_csv(path)

        ### JUST FOR TESTING - filter to beataml data
        #print(self.drug.head())
        #self.drug = self.drug[self.drug.id_type != 'DepMap_ID']
        #print(self.drug.head())
        ######################

        self.toensembl = toensembl
        self.holder = dict()
        self.start = 0
        self.failures = []

    def get_targets(self, gene_order):
        '''
        gene_order <list> - ensembl gene ids, in the order to have target (boolean) listed

        make this a generator: yield (targets, id, data source, response type, response)
        '''
        for drugname, tHGNC, tENTREZ, response, response_type, id, id_type in self.drug.values[self.start:, :]:
            tHGNC = tHGNC.strip()
            try:
                if tHGNC in self.holder:
                    targets = self.holder[tHGNC]
                else:
                    if (id_type == 'DepMap_ID'):
                        targets = [1*(self.convert_HGNC_to_ENSEMBL(tHGNC)==g) for g in gene_order]
                    else:
                        ## beatAML targets (may be multiple)
                        if ';' in tHGNC:
                            targets = tHGNC.split(';')
                        else:
                            targets = [tHGNC]
                        targets = [1 if g in [self.convert_HGNC_to_ENSEMBL(t) for t in targets] else 0 for g in gene_order]

                    self.holder[tHGNC] = targets
                assert np.sum(targets) > 0, 'target vector is all zeros'

                yield (targets, id, id_type, response_type, response)
            except AssertionError:
                self.failures.append(id)
                pass

    def convert_HGNC_to_ENSEMBL(self, gene):
        '''
        '''
        try:
            return self.toensembl[gene]
        except KeyError:
            return None


class file_name_tracker:

    def __init__ (self, label_dict=None):
        '''
        start <int> - unique identifier; for use if appending to current data sets
        '''
        if label_dict is not None:
            self.incr = max(label_dict.keys())
            self.label_dict = label_dict
        else:
            self.incr = 0
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
    toensembl = dict()
    for _,entrez,HGNC,ensembl in idmap.values:
        #print(gene)
        toensembl[HGNC] = ensembl

    return toensembl

if __name__ == '__main__':
    '''

    '''
    unpack_data2()

    # check for label_dict and load in
    LOAD_DICT_AND_CONT = False
    label_dict=None
    if os.path.exists('../data_pytorch/label_dict.pkl') and LOAD_DICT_AND_CONT:
        print('loading label_dict from file and continuing from last file...')
        with open('../data_pytorch/label_dict.pkl', 'rb') as f:
            label_dict = pickle.load(f)

    order = pd.read_csv('../data2/aml_genes_to_use.csv')['x'].values.tolist() ## load gene order (genes_to_use)
    ngenes = len(order)
    obs_size = (ngenes, 2)

    toensembl = make_toensembl('../data2/gene_id_map.csv')

    expr = expr_data(depmap_path = '../data2/depmap_expr_amlgenes.csv', beataml_path = '../data2/beataml_expr_amlgenes.csv', gene_order = order)
    drug = drug_data('../data2/drug_data_aml_genes.csv', toensembl)
    namer = file_name_tracker(label_dict=label_dict)

    try:
        ii = 0
        for targets, id, id_type, response_type, response in drug.get_targets(order):
            print(f'working ... count={ii} [failed={len(drug.failures) + len(expr.failures)}]\t', end='\r')
            obs = np.zeros(obs_size)
            obs[:,0] = expr.get_expr(id=id, source=id_type, gene_order=order)
            obs[:,1] = targets
            if np.sum(obs[:,0]) > 0 and np.sum(obs[:,1]) > 0:
                obs_name = namer.get_name(id, id_type, response_type, response)
                pyt = torch.Tensor(obs)
                torch.save(pyt, f'../data_pytorch/tensors/{obs_name}.pt')
                ii += 1
            #if ii == 5: break

        label_dict = namer.get_label_dict()
        f = open("../data_pytorch/label_dict.pkl","wb")
        pickle.dump(label_dict,f)
        f.close()
    except:
        label_dict = namer.get_label_dict()
        f = open("../data_pytorch/label_dict.pkl","wb")
        pickle.dump(label_dict,f)
        f.close()
        raise

    print()
    print(f'expression op failures: {len(expr.failures)}')
    print(f'target op failures: {len(drug.failures)}')
