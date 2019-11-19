'''

'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sbn

def discretize(embedding, image_size=(512,512)):
    '''
    embedding <dataframe>: columns [gene_id, X1, X2]
    '''
    y,x = image_size[0]-1, image_size[1]-1

    # Todo: remove outliers

    X1_pos = (embedding.x_0.values - np.amin(embedding.x_0.values))
    X2_pos = (embedding.x_1.values - np.amin(embedding.x_1.values))

    X1_norm = X1_pos / np.amax(X1_pos)
    X2_norm = X2_pos / np.amax(X2_pos)

    X1_scaled = X1_norm * y
    X2_scaled = X2_norm * x

    X1_discrete = np.around(X1_scaled, decimals=0)
    X2_discrete = np.around(X2_scaled, decimals=0)

    embedding = embedding.assign(YX = [(int(yy),int(xx)) for yy,xx in zip(X1_discrete, X2_discrete)])

    return embedding

if __name__ == '__main__':

    image_size = (256,256)
    vertices = 20000
    x_scale = 20
    y_scale = 5
    data = {'id' : np.arange(0,vertices),
            'X1' : y_scale*np.random.rand(vertices),
            'X2' : x_scale*np.random.rand(vertices),
            'expr': np.random.rand(vertices),
            'taget': np.random.choice([0,1], size=vertices, p=[0.9,0.1])}

    df = pd.DataFrame(data)

    #plt.figure()
    #sbn.scatterplot(x='X2', y='X1', data=df)
    #plt.show()

    discrete_embedding = discretize(df, image_size=image_size)

    img = np.zeros((*image_size,3))

    for id, x1, x2, expr, targ, yx in discrete_embedding.values:
        y,x = yx
        img[ y,x ,:] = np.array([expr, targ, 0])

    plt.figure()
    plt.imshow(img)
    plt.show()
