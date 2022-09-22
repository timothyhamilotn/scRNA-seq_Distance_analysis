import scanpy as sc
import numpy as np
from numpy import linalg as LA
import sklearn.metrics
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import adjusted_rand_score
import scipy
import matplotlib.pyplot as plt
import networkx as nx
import community
import seaborn as sns

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
ground_truth= pd.read_csv("Trial_2_Pancreas_original_clustering.csv",index_col=0)
g_cluster= ground_truth.iloc[:,0].values.tolist()
p_vals=[1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.2,2.4,2.6,2.8,3.0,3.25,3.5,3.75,4.0,4.5,5,5.5,6.0]
num_types= len(list(set(g_cluster)))
labels= ["Raw","PCA"]
holder_array= np.zeros((len(p_vals), 2))
for j in range(len(p_vals)):
    for i,label in enumerate(labels):
        clust_data= pd.read_csv(label+'_Minkowski_Louvain_clusters.csv',index_col=0)
        clust_mat=clust_data.values
        ARIs=len(list(set(clust_mat[:,j].tolist())))#len(list(set()))#adjusted_rand_score(g_cluster, clust_mat[:,j])
        holder_array[j,i]=ARIs

fig, axs = plt.subplots(nrows=1, ncols=1)
fig.suptitle("Num_Clusters")

#axs.set_aspect('equal')
axs.plot(p_vals,holder_array[:,0],'b', label = labels[0])
axs.plot(p_vals,holder_array[:,1], 'r', label = labels[1] )
axs.plot(p_vals,[num_types]*len(p_vals), 'g', label = "Biological Annotations" )

axs.set_xlabel('Minkowski -P')
axs.set_ylabel('Number of Clusters')
axs.legend()
    #h=ax.hist2d(new_degree,new_homog,norm=mpl.colors.LogNorm(),bins =50)
    #fig.colorbar(h[3], ax= ax)
fig.tight_layout()
plt.savefig("Num_Clusters.png")
plt.savefig("Num_Clusters.eps")

fig.clear()



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
