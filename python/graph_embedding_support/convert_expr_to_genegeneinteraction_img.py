'''

'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from continuous_2D_to_discrete import discretize
import mygene

AML_EXPR_PATH = r'C:\Users\Nate\Documents\DEEP_DRUG_SH\data\processed\beataml_expr.csv' #1.2 GB
DEPMAP_EXPR_PATH = r'C:\Users\Nate\Documents\DEEP_DRUG_SH\data\processed\depmap_expression.csv' # 0.8 GB

EMBEDDING_PATH = r'C:\Users\Nate\Documents\DEEP_DRUG_SH\data\processed\embedding_90Q.csv' #

TARGET_RESP_PATH = r'C:\Users\Nate\Documents\DEEP_DRUG_SH\data\processed\all_resp_data.csv' # 1.8 GB

NODE_ID_MAPPING = r'C:\Users\Nate\Documents\DEEP_DRUG_SH\data\processed\node_id_mapping.csv'

ENTREZ_TO_ENSEMBL_PATH = r'C:\Users\Nate\Documents\DEEP_DRUG_SH\data\processed\entrez_to_ensembl_mapping.csv'

### (x,y)
IMAGE_SIZE = (512,512)
CHANNELS = 2

class expr_to_256_val:
    def __init__(self, expr):

        self.min_ = np.amin(expr)
        self.max_ = np.quantile(expr, 0.98)#np.amax(expr)
    def scale_to_256(self, val):
        if val > self.max_:
            val = self.max_
        return 255*(val - self.min_) / (self.max_ - self.min_)

if __name__ == '__main__':
    ## read in data
    embedding = pd.read_csv(EMBEDDING_PATH)
    #aml_expr = pd.read_csv(AML_EXPR_PATH)
    depmap_expr = pd.read_csv(DEPMAP_EXPR_PATH)
    #target_resp = pd.read_csv(TARGET_RESP_PATH)
    node_to_gene_mapping = pd.read_csv(NODE_ID_MAPPING)

    print(embedding.head())
    #print(aml_expr.head())
    print(depmap_expr.head())
    #print(target_resp.head())
    print(node_to_gene_mapping.head())

    # format node mapping as dictionary
    print('making entrez -> node map')
    node_map = dict()
    for name,id in node_to_gene_mapping.values:
        node_map[name]=id


    ############################################################################
    ### This takes forever, so only run it if you have to, otherwise save it ###
    ### to disk and reuse.                                                   ###
    ###    If there are multiple ensembl mappings, we take the first one     ###
    ###    Some entrez id's don't have an ensembl id:                        ###
    ###
    ############################################################################
    # '''
    # print('retreiving entrez->ensembl mapping')
    # mg = mygene.MyGeneInfo()
    # entrez_ids = depmap_expr.entrez_id.unique().tolist()
    # entrez_to_ensembl_map = dict()
    #
    # #f = open(r'C:\Users\Nate\Documents\DEEP_DRUG_SH\data\processed\entrez_to_ensembl_mapping.csv', 'w')
    # f.write('entrez_id, ensembl_id\n')
    # for i,ent in enumerate(entrez_ids):
    #     print(f'complete: {100*i/len(entrez_ids):.2f}% [{i}]', end='\r')
    #     try:
    #         ensembl = mg.getgene(ent, fields='ensembl.gene',email='evansna@ohsu.edu')['ensembl']
    #         if type(ensembl) == type([]):
    #             ensembl = ensembl[0]['gene']
    #         else:
    #             ensembl = ensembl['gene']
    #         entrez_to_ensembl_map[ent] = ensembl
    #         f.write(f'{ent},{ensembl}\n')
    #     except:
    #         print(f'failed: {ent}', end='\t\t\t\n')
    #         #raise
    # f.close()
    # '''
    ############################################################################
    ###                                                                      ###
    ###                                                                      ###
    ############################################################################
    print('loading entrez -> ensembl id mapping from disk...')
    temp = pd.read_csv(ENTREZ_TO_ENSEMBL_PATH)
    entrez_ensembl_map = dict()
    for ent, ens in temp.values:
        entrez_ensembl_map[ent] = ens

    ## discretize the embedding path
    embedding = discretize(embedding, image_size=IMAGE_SIZE)

    print(embedding.head())

    print(f'number of vertices: {len(embedding.id)}')
    print(f'number of unique pixel mappings: {len(embedding.YX.unique())}')
    print(f'number of overlapping vertice -> pixel mappings: {len(embedding.id)-len(embedding.YX.unique())}')

    cell_line_ids = depmap_expr.cell_line.unique()

    expr_imgs = dict()
    failed = set()
    converter = expr_to_256_val(depmap_expr.expression.values)

    for line in cell_line_ids:
        expr = depmap_expr[depmap_expr.cell_line == line]
        print(f'length of single obs expr set: {expr.values.shape[0]}', end='\t\t\t\n')
        expr_imgs[line] = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], CHANNELS)) # 3 channels for now to make it easy to visualize
        i = 0
        for hgnc, entrez_id, cl, expr_val in expr.values:
            #print(f'cell line: {line} \t| gene: {hgnc} \t| count: {i}', end='\t\t\r')
            print(f'processing "{line}": {100*i/19000:.2f}%', end='\r')
            i+=1
            try:
                ensembl_id = entrez_ensembl_map[entrez_id]
                coord = embedding[embedding.id == node_map[ensembl_id]].YX.values[0]
                #### If we have duplicate mappings, we need to average over them
                if expr_imgs[line][coord[0], coord[1], 0] != 0:
                    expr_imgs[line][coord[0], coord[1], 0] = np.average([expr_imgs[line][coord[0], coord[1], 0], converter.scale_to_256(expr_val)])
                else:
                    expr_imgs[line][coord[0], coord[1], 0] = np.array([converter.scale_to_256(expr_val)])
            except:
                #print()
                #print(f'failed: {line}-{hgnc}')
                failed.add(entrez_id)
                #raise

        print()
        print(f'number of genes failed to map into embedding: {len(failed)}')
        plt.figure(figsize=(12,8))
        #plt.imshow(expr_imgs[line][:,:,0].reshape(IMAGE_SIZE), vmin=0, vmax=255)
        plt.imsave(fname=f'C:\\Users\\Nate\\Documents\\DEEP_DRUG_SH\\output\\{line}.png', arr=expr_imgs[line][:,:,0].reshape(IMAGE_SIZE), vmin=0, vmax=255)




    ## for each cell_line, generate an expr image, but leave target region blank
    ## save in a dictionary as cell_line_id -> expr_image_ndarray
            ## discretize embedding
            ## use mygene to convert gene_ids and the id_mapping to insert into embedded space

    ## for each target, response pair
        ## insert target into target channel of gene image
        ## generate obs_id for target,response,cell_line
        ## make label_dict : obs_id -> Resp (AUC/Dep)

    ## save the expr_image as a pytorch tensor file

    ## make a dataloader for each datasets
