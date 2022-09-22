import scanpy as sc
import numpy as np
from numpy import linalg as LA
import sklearn.metrics
from sklearn.neighbors import DistanceMetric
import scipy
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.neighbors import NearestNeighbors
#import pickle
import pandas as pd
import joblib as jb

def neighbors(data, k=20):
    # for a given dataset, finds the k nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k, metric= "precomputed").fit(data)
    distances, indices = nbrs.kneighbors()
    graphs=nbrs.kneighbors_graph()

    return indices,graphs

print("Doing Preprocessing")
r = sc.read_h5ad('Panc_Data.h5ad')
new_r=  r[r.obs['tech'].isin(['celseq2'])]
barcodes= new_r.obs.index.tolist()
clust_list = [i for i in list((new_r.obs["celltype"]))]
cell_types= pd.DataFrame({"Cell_Type": clust_list}, index= barcodes)
cell_types.to_csv("Trial_2_Pancreas_original_clustering.csv")
#sc.pp.neighbors(new_data,n_neighbors = 20,use_rep = 'X')
#sc.tl.louvain(new_data,resolution = 0.5)
sc.pp.normalize_per_cell(new_r, counts_per_cell_after=1e4)
sc.pp.log1p(new_r)
new_r.raw = new_r
sc.pp.highly_variable_genes(new_r,subset = True,n_top_genes = 2000)
sc.pp.scale(new_r, max_value=10)
sc.tl.pca(new_r, n_comps= 50)
Loading_Scores= new_r.obsm['X_pca']
column_names = ["PC_"+str(i+1) for i in range(Loading_Scores.shape[1])]
PCs= pd.DataFrame(Loading_Scores, columns= column_names, index= barcodes)
PCs.to_csv("Trial_2_Pancreas_PCA_Coordinates.csv")
sc.tl.tsne(new_r,n_pcs=8)
TSNE_Coords= new_r.obsm['X_tsne']
TSNE_D= pd.DataFrame(TSNE_Coords, columns= ["TSNE_"+str(i+1) for i in range(TSNE_Coords.shape[1])], index= barcodes)
TSNE_D.to_csv("Trial_2_Pancreas_TSNE_Coordinates.csv")
sc.pl.tsne(new_r ,color = ['celltype'], save= "Pancreas_Type_2_tsne_plot.png")

#G7tt6Et2
#Cho1ces4me
#a = np.arange(9) - 4
#b = a.reshape((3, 3))

#t2 = sklearn.metrics.pairwise.pairwise_distances(q.layers['counts'][:1000], Y=None, metric='minkowski', n_jobs=6, p = 2)
#Z = linkage(np.mat(t2), 'average')
#fig = plt.figure(figsize=(25, 10))
#dn = dendrogram(Z)
#partition = fcluster(Z, t=2, criterion='maxclust')

print('beginning')
p_vals=[1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.2,2.4,2.6,2.8,3.0,3.25,3.5,3.75,4.0,4.5,5,5.5,6.0]


for cl in p_vals:
    for data,label in zip([new_r.layers['counts'][:],Loading_Scores],["Raw","PCA"]):
        leed = sklearn.metrics.pairwise.pairwise_distances(data, Y=None, metric='minkowski', n_jobs=-2,p=cl)
        print(leed.shape)
        neigh,graph= neighbors(leed)
        df = pd.DataFrame(data=leed)
        neigh_df= pd.DataFrame(data=neigh)
        df.to_csv(label+'_Trial_2_Dist/Pancreas_scrna_p_'+str(cl).replace('.','_')+'.csv', sep=',', header=False, float_format='%.2f', index=False)
        neigh_df.to_csv(label+'_Trial_2_kNN/Pancreas_scrna_p_'+str(cl).replace('.','_')+'.csv', sep=',', header=False, float_format='%.2f', index=False)
        #jb.dump(graph,label+'_kNN_Graph/Pancreas_scrna_p_'+str(cl).replace('.','_')+".pkl")
    print(cl)


'''
b = np.random.rand(3,3)
print(b)
for n in range(1,50):
    p = n/10.0
    #print(LA.norm(b[1]+b[2],p),LA.norm(b[0]+b[2],p),LA.norm(b[1]+b[0],p))
    q = [LA.norm(b[1]-b[2],p),LA.norm(b[0]-b[2],p),LA.norm(b[1]-b[0],p)]
    #print(p)
    #print(q)
    print(p,np.argsort(q))
    print(q)

'''
