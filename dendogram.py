# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 11:56:57 2022

@author: marcs
"""
import numpy as np

def euclid(v1, v2): #Function to calculate eulcidian distance between two vectors 
    return np.linalg.norm(v1-v2)

def single_linkage(c1, c2):
    min_dist = np.inf
    
    for i in range(c1.shape[0]):
        for j in range(c2.shape[0]):
            dist = euclid(c1[i], c2[j])
            if dist < min_dist: # check if current distnce is lower than current minimum
                min_dist = dist
    
    return min_dist
            

def complete_linkage(c1, c2):
    max_dist = 0
    
    for i in range(c1.shape[0]):
        for j in range(c2.shape[0]):
            dist = euclid(c1[i], c2[j])
            if dist > max_dist: # check if current distance is higher than current maximum 
                max_dist = dist
    
    return max_dist
    

def centroid_linkage(c1, c2):
    c1_centriod = np.mean(c1, axis = 0) #Find the centriods of both clusters 
    c2_centriod = np.mean(c2, axis = 0)
            
    return euclid(c1_centriod, c2_centriod)


def average_linkage(c1, c2):
    all_dist = np.zeros((c1.shape[0], c2.shape[0]))
    
    for i in range(c1.shape[0]):
        for j in range(c2.shape[0]):
            all_dist[i,j] = euclid(c1[i], c2[j])
    
    return np.mean(all_dist)

class denogram:
    
    
    def __init__(self, clusters, left_node = None, right_node = None, height = 0.0, is_leaf = False):
        
        """
        Intialises denogram object
        -----------------------------------
        
        clusters: list of array's (expect in case where leaf = True then multi-dimensional array)
        left_node / right_node: If not leaf an array containg the samples in the left / right child node
        Height: a float containg the height of the current node
        is_leaf: True or False
        
        """
        
        
        if len(clusters) == 1:
            self.is_root = True
        else:
            self.is_root = False
        
        self.clusters = clusters
        self.is_leaf = is_leaf
        self.height = height
        self.left_node = left_node
        self.right_node = right_node
        self.no_clusters = len(clusters)
        
        
    def newnode(self, link_function):
        no_clusters = self.no_clusters
        clusters = self.clusters
        min_dist = np.inf
       #all_dist = np.zeros((no_clusters, no_clusters))
        
        for i in range(no_clusters):
            for j in range(no_clusters):
                
                if i < j:
                    dist = link_function(clusters[i], clusters[j])
                else:
                    dist = np.inf
                
                if dist < min_dist: # Find the minumium distnce based on the linake function
                    min_dist = dist
                    LN = i
                    RN = j
                #all_dist[i, j] = dist
        #print(all_dist)
        
        new_cluster = np.concatenate((clusters[LN], clusters[RN]), axis = 0)
        new_right_node = clusters[RN]
        new_left_node = clusters[LN]
        
        new_clusters = [] #Create empty list to store all new clusters
        new_clusters.append(new_cluster) # add the new cluster to new clusters 

        for i in range(no_clusters):
            if i != RN and i != LN:
                new_clusters.append(clusters[i]) # add all clusters to new_clusters apart from the two child nodes
        
        
        return denogram(new_clusters, left_node = new_left_node, right_node= new_right_node, height= min_dist )
        
        
def createDenogram(data, link_function):
    
    deno = [] # list to store all denogram objects
    deno.append(denogram(data, is_leaf= True))
    
    for i in range(len(data) - 1):
        deno.append(deno[-1].newnode(link_function)) #add the new node object to end of list
    
    return deno
    
                
    
#%%
#Test on the toy example on lecture slides 

test_data = [np.array([[1,4]]), np.array([[6, 4]]), np.array([[0, 3]]), np.array([[1, 3]]), np.array([[5, 6]]), np.array([[5, 1]]), np.array([[6, 2]])]

iter0 = denogram(test_data, is_leaf= True)

iter1 = iter0.newnode(complete_linkage)
iter2 = iter1.newnode(complete_linkage)
iter3 = iter2.newnode(complete_linkage)
iter4 = iter3.newnode(complete_linkage)
iter5 = iter4.newnode(complete_linkage)
iter6 = iter5.newnode(complete_linkage)

test_denogram = createDenogram(test_data, complete_linkage)

#%%
#In this cell i will import the nci dataset and get it into the correct format for my denogram class
#The required format to intialise an instance of the denogram class is a list of arrays containg each sample
#Each array must be of the form 1 by number of atrributes

data = np.genfromtxt('ncidata.txt')
data = np.transpose(data) # transpose so that every row represents a sample

nci_data = []

for i in range(data.shape[0]):
    nci_data.append(np.expand_dims(data[i],axis = 0))

#%%
##PART A 
#Get the data structures representing denograms using single linkage, complete linkage, average linkage and centroid linkage.

denogram_single = createDenogram(nci_data, single_linkage)
denogram_complete = createDenogram(nci_data, complete_linkage)
denogram_centriod = createDenogram(nci_data, centroid_linkage)
denogram_average = createDenogram(nci_data, average_linkage)

#%%
##PART B
#Create function get clusters 

def get_clusters(K, Denogram):
    """
    

    Parameters
    ----------
    K : int
        The number of clusters desired.
    Denogram : List of denogram objects
        The denogram desired 

    Returns
    -------
    List of 2D arrays's:
        Each array contains a cluster. Every row within each array contains a specific sample.
    


    """
    
    for i in range(len(Denogram)): # search through denogram to find the node with K clusters
        if Denogram[i].no_clusters == K:
            K_node = Denogram[i]
            break
    
    return K_node.clusters

#%%
#PART C
# The performance of a particular linkage function can be evaluated by the sum of the euclidin ditance squared beteen each observation
# and it's clusters centriod. A smaller sum suggests a better cluster.

# First i will create a function which computes the sum of the distances squared between the cluster centriod and each observation 
# for every iteration of AHC (every node in the denogram). 

def cluster_score(Denogram):
    """
    

    Parameters
    ----------
    Denogram : List of denogram objects 
        Every denogram object is a node of the denogram.

    Returns
    -------
    Gives sum(||x - v||^2), where x is an observation in the cluster and v is the cluster centriod.
    This is done for every iteration of AHC/node in denogram. 
    The sum for each iteration is outputed in an array. 

    """
    no_iter = len(Denogram) - 1 # -1 as first object has every iteration in its own cluster so the sum = 0 and not intersting
    iter_sum = np.zeros(no_iter)
    
    for i in range(no_iter):
        current_clusters = Denogram[i + 1].clusters # + 1 to skip first node. current_clustrs stores all clusters in node
        current_sum = 0 # store sum(||x - v||^2) for current iteration/node in AHC
        
        for j in range(len(current_clusters)):
            cluster = current_clusters[j] #Get a cluster
            cluster_centroid = np.mean(cluster, axis = 0) #Find it's centriod
            
            for z in range(len(cluster)):
                current_sum += np.square(euclid(cluster_centroid, cluster[z])) # Find ||x - v||^2
    
        iter_sum[i] = current_sum #Store sum(||x - v||^2) for the current iteration/node
    
    return iter_sum

#%%
# Now a function to calculate sum(||x - v||^2) for evry iteration has been defined, the function is used to calculate sum(||x - v||^2)
# for each denogram


single_performance = cluster_score(denogram_single)
complete_performance = cluster_score(denogram_complete)
centriod_performance = cluster_score(denogram_centriod)
average_performance = cluster_score(denogram_average)

#%%
#Now plot these for each iteration
import matplotlib.pyplot as plt

iteration = np.arange(1, len(single_performance) + 1)

plt.plot(iteration, single_performance, color = 'midnightblue')
plt.plot(iteration, complete_performance, color = 'springgreen')
plt.plot(iteration, centriod_performance, color = 'orange')
plt.plot(iteration, average_performance, color = 'lightcoral')
plt.xlabel('iteration')
plt.ylabel('sum(||x - v||^2)')
plt.legend(['single', 'complete', 'centriod', 'average'])
plt.title('Performance of linkage functions')



