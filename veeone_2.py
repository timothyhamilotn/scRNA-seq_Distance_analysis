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
barcodes= r.obs.index.tolist()
clust_list = [i for i in list((r.obs["celltype"]))]
cell_types= pd.DataFrame({"Cell_Type": clust_list}, index= barcodes)
cell_types.to_csv("Pancreas_original_clustering.csv")
tech_types= r.obs['tech'].values.tolist()
tech_frame= pd.DataFrame({"Technical_Type": tech_types}, index= barcodes)
tech_frame.to_csv("Pancreas_technical_type.csv")
#sc.pp.neighbors(new_data,n_neighbors = 20,use_rep = 'X')
#sc.tl.louvain(new_data,resolution = 0.5)
sc.pp.normalize_per_cell(r, counts_per_cell_after=1e4)
sc.pp.log1p(r)
r.raw = r
sc.pp.highly_variable_genes(r,subset = True,n_top_genes = 2000)
sc.pp.scale(r, max_value=10)
sc.tl.pca(r, n_comps= 50)
Loading_Scores= r.obsm['X_pca']
column_names = ["PC_"+str(i+1) for i in range(Loading_Scores.shape[1])]
PCs= pd.DataFrame(Loading_Scores, columns= column_names, index= barcodes)
PCs.to_csv("Pancreas_PCA_Coordinates.csv")
sc.tl.tsne(r,n_pcs=8)
TSNE_Coords= r.obsm['X_tsne']
TSNE_D= pd.DataFrame(TSNE_Coords, columns= ["TSNE_"+str(i+1) for i in range(TSNE_Coords.shape[1])], index= barcodes)
TSNE_D.to_csv("Pancreas_TSNE_Coordinates.csv")
sc.pl.tsne(r ,color = ['celltype'], save= "Pancreas_tsne_plot.png")
sc.pl.tsne(r ,color = ['tech'], save= "Pancreas_tsne_plot_tech.png")
#G7tt6Et2
#Cho1ces4me
#a = np.arange(9) - 4
#b = a.reshape((3, 3))

#t2 = sklearn.metrics.pairwise.pairwise_distances(q.layers['counts'][:1000], Y=None, metric='minkowski', n_jobs=6, p = 2)
#Z = linkage(np.mat(t2), 'average')
#fig = plt.figure(figsize=(25, 10))
#dn = dendrogram(Z)
#partition = fcluster(Z, t=2, criterion='maxclust')



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
