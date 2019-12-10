'''
DEPRECATED - just use the FPKM data from DepMap, no need to recalculate 

This script is used to extract and combine all the STAR-counts beatAML expression data that has been download from NCI GDC.

Expects all data to be in a single folder with no spurrious files or directories

BeatAML project page
https://portal.gdc.cancer.gov/projects/BEATAML1.0-COHORT

Relevant Expr Files (to build manifest from)
https://portal.gdc.cancer.gov/repository?facetTab=files&filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.project_id%22%2C%22value%22%3A%5B%22BEATAML1.0-COHORT%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.analysis.workflow_type%22%2C%22value%22%3A%5B%22STAR%20-%20Counts%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_type%22%2C%22value%22%3A%5B%22Gene%20Expression%20Quantification%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.experimental_strategy%22%2C%22value%22%3A%5B%22RNA-Seq%22%5D%7D%7D%5D%7D&searchTableTab=files

To run:
$ python extract_beatAML_expr.py /path/to/data/dir /path/to/output/dir
'''
import sys
import os
import gzip
import shutil
import pandas as pd
import json
import requests
import numpy as np

if __name__ == '__main__':

    _, path_in, path_out = sys.argv

    data = None
    fail = []
    UUIDS=[]
    nfiles = len(os.listdir(path_in))
    for i,dir in enumerate(os.listdir(path_in)):
        print('processing beatAML data [%d/%d]' %(i,nfiles), end='\r')
        try:
            file_in = os.path.abspath( path_in + '/' + dir + '/' + [x for x in os.listdir('%s%s' %(path_in, dir)) if x[-3:] == '.gz'][0] )
            UUIDS.append(dir)
            file_out = file_in[0:-3]
            ####################################################################
            # unzip .gz files
            ####################################################################
            with gzip.open(file_in, 'rb') as f_in:
                with open(file_out, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            ####################################################################
            # Retrieve entity id
            ####################################################################
            url = 'https://api.gdc.cancer.gov/files/%s?pretty=true&expand=cases.samples' %dir
            r = requests.get(url)
            sample_type = r.json()['data']['cases'][0]['samples'][0]["sample_type"]
            entity_id = r.json()['data']['cases'][0]['samples'][0]["submitter_id"]
            #print()
            #print('entity id: %s' %entity_id)
            #print('sample type: %s' %sample_type)
            ####################################################################

            ####################################################################
            # load uncompressed data
            ####################################################################
            data2 = pd.read_csv(file_out,sep='\t', header=None)
            genes = data2.values[:,0]
            expr = data2.values[:,1]

            ####################################################################
            ### Normalize data
            ####################################################################
            # to match DepMap Expr (TPM;not sure how to match this) log2 transformed
            # data with pseudocount of 1

            log2_FPKM = np.array([np.log2((x+1)) for x in (expr)]) # weird numpy error otherwise
            D = {gene: [expr] for gene,expr in zip(genes, log2_FPKM)}
            data2 = pd.DataFrame(D).assign(UUID=dir, entity_id=entity_id, sample_type=sample_type)[['entity_id', 'UUID', 'sample_type'] + genes.tolist()]
            ####################################################################

            if data is None:
                data = data2
            else:
                data = data.append(data2, ignore_index=True)
        except:
            fail.append(dir)
            raise

    print()
    #print(data.values[0:5,0:5])
    print('processing Failures (%d): %s' %(len(fail), str(["\t%s\n" %x for x in fail])))

    with open(path_out, 'w') as f:
        f.write(data.to_csv(index=False))

    print('data saved to:  %s' %path_out)
