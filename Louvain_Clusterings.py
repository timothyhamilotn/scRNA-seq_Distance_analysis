import scanpy as sc
import numpy as np
from numpy import linalg as LA
import sklearn.metrics
from sklearn.neighbors import DistanceMetric
import scipy
import matplotlib.pyplot as plt
import networkx as nx
import community

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.neighbors import NearestNeighbors
#import pickle
import pandas as pd
import joblib as jb
import pickle

def neighbors(data, k=20):
    # for a given dataset, finds the k nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k, metric= "precomputed").fit(data)
    distances, indices = nbrs.kneighbors()
    graphs=nbrs.kneighbors_graph()

    return indices,graphs



#G7tt6Et2
#Cho1ces4me
#a = np.arange(9) - 4
#b = a.reshape((3, 3))

#t2 = sklearn.metrics.pairwise.pairwise_distances(q.layers['counts'][:1000], Y=None, metric='minkowski', n_jobs=6, p = 2)
#Z = linkage(np.mat(t2), 'average')
#fig = plt.figure(figsize=(25, 10))
#dn = dendrogram(Z)
#partition = fcluster(Z, t=2, criterion='maxclust')

p_vals=[1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.2,2.4,2.6,2.8,3.0,3.25,3.5,3.75,4.0,4.5,5,5.5,6.0]

for label in ["Raw","PCA"]:
    print(label)
    clusterings = []
    for cl in p_vals:
        dist_mat=pd.read_csv(label+'_Trial_2_Dist/Pancreas_scrna_p_'+str(cl).replace('.','_')+'.csv', sep=',', header=None, index_col= False)
        print(dist_mat.head)
        print(dist_mat.shape)
        ind,graph= neighbors(dist_mat.values)
        nx_graph= nx.from_scipy_sparse_matrix(graph)
        partition = community.best_partition(nx_graph)
        clusterings.append(list(partition.values()))
        print(cl)
        print(len(list(partition.values())))
    new_array= np.asarray(clusterings)
    new_array= np.transpose(new_array)
    print("Saving to file")
    print(new_array.shape)
    new_frame= pd.DataFrame(data= new_array, columns= p_vals)
    new_frame.to_csv(label+'_Minkowski_Louvain_clusters.csv')



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
