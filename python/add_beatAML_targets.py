'''
Add drug targets to beatAML drugs
'''

import pandas as pd
import numpy as np
import difflib as dl

def get_closest_synonym(name, syns):

    dl.get_close_matches(name, )

if __name__ == '__main__':

    targets = pd.read_csv('./../data/targetome/Targetome_FullEvidence_011019.txt', sep='\t')
    synonyms = pd.read_csv('./../data/targetome/Targetome_DrugSynonymDictionary_100917.txt', sep='\t')
    beatAML = pd.read_csv('./../data/beatAML_waves1_2/OHSU_BeatAMLWaves1_2_Tyner_DrugResponse.txt', sep='\t')

    drugs = beatAML[['inhibitor']].drop_duplicates()
    print(drugs)

    drugs = drugs.assign(Synonym=[dl.get_close_matches(x, synonyms.Synonym, n=1, cutoff=0.0)[0] for x in drugs.inhibitor]).merge(synonyms, on='Synonym', how='left')[['inhibitor','Synonym','Drug']]

    print(drugs.head(10))

    with open('./drugs.csv', 'w') as f:
        f.write(drugs.to_csv(index=False))
