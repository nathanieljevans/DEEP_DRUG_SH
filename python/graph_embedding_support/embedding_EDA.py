'''

'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sbn

if __name__ == '__main__':

    #embedding = pd.read_csv(r'C:\Users\Nate\Documents\AttentionWalk\output\chameleon_AW_embedding.csv')
    embedding = pd.read_csv(r'C:\Users\Nate\Documents\DEEP_DRUG_SH\gene_network_embedding\embedding_all_edges.csv')

    plt.figure()
    sbn.scatterplot(x='x_0', y='x_1', alpha=0.1, data=embedding)
    plt.show()
