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

def jaccard(A,B):
    # for two sets A and B, finds the Jaccard distance J between A and B
    A = set(A)
    B = set(B)
    union = list(A|B)
    intersection = list(A & B)
    J = ((len(union) - len(intersection))/(len(union)))
    return(J)
def find_AJD(A,B):
    holder = 0
    for i in range(A.shape[0]):
        holder+= jaccard(A[i,:], B[i,:])
    return holder/A.shape[0]

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
holder_array= np.zeros((len(p_vals),len(p_vals)))
    #holder_array= np.zeros((len(p_vals),len(p_vals)))
for a,i in enumerate(p_vals):
    neigh_one = pd.read_csv('Raw_Trial_2_kNN/Pancreas_scrna_p_'+str(i).replace('.','_')+'.csv',index_col = None, header = None)
    print(neigh_one.head())
    for b,j in enumerate(p_vals):
        neigh_two= pd.read_csv('PCA_Trial_2_kNN/Pancreas_scrna_p_'+str(j).replace('.','_')+'.csv',index_col = None, header = None)
        ARIs=find_AJD(neigh_one.values, neigh_two.values)
        holder_array[a,b]=ARIs
        #holder_array[b,a]=ARIs
new_frame= pd.DataFrame(holder_array, index= p_vals, columns= p_vals)
new_frame.to_csv('Verses_Minkowski_Louvain_AJD.csv')
fig, axs = plt.subplots(nrows=1, ncols=1)
fig.suptitle("Raw Versus PCA AJD")
axs= sns.heatmap(new_frame,xticklabels="auto", yticklabels='auto', ax= axs,robust=False, cmap = "cividis")
axs.set_xlabel("PCA Neighborhoods with Specificed Miknowski-P")
axs.set_ylabel("Raw Neighborhoods with Specificed Miknowski-P")
plt.tight_layout()
plt.savefig('Verses_Minkowski_Louvain_AJD.png')
plt.savefig('Verses_Minkowski_Louvain_AJD.eps')





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
