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
for a in labels:
    for p in p_vals:
        neighborhoods= pd.read_csv(a+'_Trial_2_kNN/Pancreas_scrna_p_'+str(p).replace('.','_')+'.csv', index_col= False, header= None)
        neighborhoods = neighborhoods.values
        type_holder=[0]*neighborhoods.shape[0]
        count_holder= np.zeros((neighborhoods.shape[0],len(cell_types)))
        for i in range(neighborhoods.shape[0]):
            type_holder[i]= master_dict[i]
            for j in range(neighborhoods.shape[1]):
                c_type = master_dict[neighborhoods[i,j]]
                slot= cell_types.index(c_type)
                count_holder[i,slot]+=1
        count_holder = (count_holder/20)*100
        count_frame= pd.DataFrame(count_holder,columns=cell_types)
        count_frame.insert(0,"Cell_Type", type_holder, True)
        count_frame=count_frame.sort_values(by = "Cell_Type")
        c_data=count_frame.iloc[:,1:]
        #count_frame.insert(1,"Cluster_ID", clust_list,True)
        #count_frame.to_csv("/media/timothyhamilton/data1/Tim_Hamilton/Zheng_DDG/Results/metric_"+str(p_val)+"_heatmap"+dataset+"_mini_merge_"+gene_set+"_"+str(num_neighbors)+"_"+normalization+"_"+rand_rap+".csv")
        colors= sns.color_palette("husl", len(cell_types))
        color_dict = dict(list(zip(cell_types,colors)))
        col_series= count_frame.iloc[:,0].map(color_dict)
        cg = sns.clustermap(c_data,cmap='bwr',row_cluster= False, col_cluster= False, row_colors = [col_series], linewidths=0, yticklabels= False)
        for l in cell_types:
            cg.ax_col_dendrogram.bar(0, 0, color=color_dict[l],label=l, linewidth=0)
        cg.ax_col_dendrogram.legend(loc="center", ncol=int(len(cell_types)/3), title = "Cell Types")
        cg.savefig(a+'_Purity/Pancreas_scrna_p_'+str(p).replace('.','_')+'.png')




#sc.pp.neighbors(adata,n_neighbors = 20,use_rep = 'X')
#sc.tl.louvain(adata,resolution = 0.5)
#clust_list = [int(i) for i in list(adata.obs["louvain"])]






    ### Write the result to .csv



### if you want to read a loom file:
# adata = sc.read_loom(filename)

# Do a nearest neighbor search:
