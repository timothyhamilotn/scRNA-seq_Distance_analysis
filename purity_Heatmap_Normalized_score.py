import pandas as pd
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import DistanceMetric

import smtplib
import os
import sys



# read the data from 10x .mtx:
def neighbors(data, k=20, pval = 2):
    # for a given dataset, finds the k nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', p= pval).fit(data)
    distances, indices = nbrs.kneighbors(data)
    return indices[:,1:]

def jaccard(A,B):
    # for two sets A and B, finds the Jaccard distance J between A and B
    A = set(A)
    B = set(B)
    union = list(A|B)
    intersection = list(A & B)
    J = ((len(union) - len(intersection))/(len(union)))
    return(J)
def equalizer(A):
    for i in range(A.shape[1]):
        if np.mean(A[:,i])>=1.0:
            A[:,i]/=np.mean(A[:,i])
    return A
ground_truth= pd.read_csv("Trial_2_Pancreas_original_clustering.csv",index_col=0)
g_clust= ground_truth.iloc[:,0].values.tolist()
master_dict= dict(zip(list(range(0,len(g_clust))), g_clust))
cell_types= list(set(g_clust))
cell_types.sort()


p_vals=[1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.2,2.4,2.6,2.8,3.0,3.25,3.5,3.75,4.0,4.5,5,5.5,6.0]
labels= ["Raw","PCA"]
holder_array= np.zeros((len(p_vals),len(labels)))
for b,a in enumerate(labels):
    for q,p in enumerate(p_vals):
        neighborhoods= pd.read_csv(a+'_Trial_2_kNN/Pancreas_scrna_p_'+str(p).replace('.','_')+'.csv', index_col= False, header= None)
        neighborhoods = neighborhoods.values
        type_holder=[0]*neighborhoods.shape[0]#np.asarray([0]*neighborhoods.shape[0])
        count_holder= [0]*neighborhoods.shape[0]
        for i in range(neighborhoods.shape[0]):
            type_holder[i]= master_dict[i]
            for j in range(neighborhoods.shape[1]):
                c_type = master_dict[neighborhoods[i,j]]
                if c_type == type_holder[i]:
                    count_holder[i]+=1
        new_frame=pd.DataFrame({"Percent_Homogenous":[100*i/20 for i in count_holder], "Type": type_holder})
        mean_holder=[0]*len(list(set(type_holder)))
        for r,p in enumerate(list(set(type_holder))):
            mean_holder[r]=np.mean(new_frame.loc[new_frame["Type"]==p,"Percent_Homogenous"].values.tolist())
        holder_array[q,b]=np.mean(mean_holder)
fig, axs = plt.subplots(nrows=1, ncols=1)
fig.suptitle(" Normalized Neighborhood Homogeneity")

#axs.set_aspect('equal')
axs.plot(p_vals,holder_array[:,0],'b', label = labels[0]+ "W/O Preprocessing")
axs.plot(p_vals,holder_array[:,1], 'r', label = labels[1]+"After Preprocessing" )

axs.set_xlabel('Minkowski -P')
axs.set_ylabel('Normalized Average Neighborhood Homogeneity')
axs.legend()
    #h=ax.hist2d(new_degree,new_homog,norm=mpl.colors.LogNorm(),bins =50)
    #fig.colorbar(h[3], ax= ax)
fig.tight_layout()
plt.savefig("Homog_Ref_Norm.png")
plt.savefig("Homog_Ref_Norm.eps")

fig.clear()

#sc.pp.neighbors(adata,n_neighbors = 20,use_rep = 'X')
#sc.tl.louvain(adata,resolution = 0.5)
#clust_list = [int(i) for i in list(adata.obs["louvain"])]






    ### Write the result to .csv



### if you want to read a loom file:
# adata = sc.read_loom(filename)

# Do a nearest neighbor search:
