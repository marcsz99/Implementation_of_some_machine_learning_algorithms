# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:58:22 2022

@author: marcs
"""
from random import sample
from random import choices
from random import seed 
import math
# in this short Python script, the dataset is assumed to contain observations of the tuple form (x_1,x_2, ... , x_p,y), where x_j's are attributes and y is the label
# "data" is a list of observations, "indx" is a list of indices to the observations




def avg(data,indx):
	if len(indx) == 0:	return 0.0
	return sum([ data[i][-1] for i in indx ]) / len(indx)

def rss(data,indx):
	if len(indx) == 0:	return 0.0
	mean = avg(data,indx)
	return sum([ pow( data[i][-1]-mean , 2.0 ) for i in indx ])

class randomforest:
	#There are two types of nodes.
	#When the node is a leaf, then self.leaf = True, and the prediction is the average of the labels of the data reaching this leaf.
	#When the node is not a leaf, then self.attr and self.split record the optimal split, and self.L, self.R are two sub-decision-trees.

    def __init__(self,data,indx,depth):						#if you do not want to limit depth, you can set depth = len(data)
        if depth==0:										#if depth = 0, that means we don't go further down the tree
            self.leaf = True								#so it is a leaf
            self.prediction = avg(data,indx)				#and the prediction is the average of all labels in data[indx]
        elif len( set([data[i][-1] for i in indx]) ) == 1:	#when all labels in data[indx] are the same, we don't need to go further down the tree
            self.leaf = True								#this includes the case when len(indx)==1; so it is a leaf
            self.prediction = data[indx[0]][-1] 			#and the prediction is simply the common label value
        #elif len(indx)== 0:                                 #In the case where there is no data in a subtree the tree can't be further splitted
           # print('NO DATA!')
           # self.leaf = True                                # so it must be a leaf. The prediction is given as the average of all labels as there is no data in the subtree
           # self.prediction = avg(data, list(range(len(data)))) # to base my predictions off. 
        else:												#otherwise, we need to do splitting; computing optimal split below
            self.leaf = False								#so it is not a leaf
            self.attr , self.split , self.L , self.R = self.generate(data,indx,depth)
			#generate is the function below, which computes the optimal split
			#attr stores which attribute is used to split
			#split stores the numerical value used to split into left and right subtrees
			#L and R are left and right subtrees

    def generate(self,data,indx,depth):

       # p = len(data[indx[0]])-1
        p = 13
        no_atrr_samp = math.ceil(p / 3)               #find the number of atrributes to sample
        sampled_atrr = sample(range(p), no_atrr_samp)      #find the sampled atrributes 
        print()
        labels = [ data[i][-1] for i in indx ]
        opt = pow ( max(labels) - min(labels) , 2.0 ) * len(indx) + 1.0
        
        for j in sampled_atrr:									#for each sampled attribute, we search the optimal split
            all_cuts = set( [ data[i][j] for i in indx ] )	#we find out all possible split values of the attribute we are considering
            for cut in all_cuts:
                #Values on the bounadry of the cut will be repated twice. 
                yl = [ i for i in indx if data[i][j] < cut ]	#yl is the list of indices to those observations where its j-th attribute value is <= cut
                yr = [ i for i in indx if data[i][j] >= cut ]	#yr is the list of indices to those observations where its j-th attribute value is > cut
                if len(yl) == 0 or len(yr) == 0:
                    yl = [ i for i in indx if data[i][j] <= cut ]	#This if statments makes sure there is at least one sample in each cut 
                    yr = [ i for i in indx if data[i][j] >= cut ]
                    
                tmp = rss(data,yl) + rss(data,yr)
                if tmp < opt:
                    opt , attr , split, L , R = tmp , j , cut , yl , yr
            return attr , split , randomforest(data,L,depth-1) , randomforest(data,R,depth-1)
		#after finding the optimal split, at each child node we recursively generate a decision tree of height depth-1

    def predict(self,x):
        if self.leaf == True:	return self.prediction
        if (x[self.attr] <= self.split):	return self.L.predict(x)
        return self.R.predict(x)


# First obtain the boston dataset 
#with open('boston2.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
 #   csv_reader = reader(read_obj)
    # Get all rows of csv from csv_reader object as list of tuples
  #  list_of_tuples = list(map(tuple, csv_reader))
    # display all rows of csv

#atrr_names = list_of_tuples[0]
#boston_data = list_of_tuples[1:]

import pandas as pd

df = pd.read_csv (r'C:\Users\marcs\OneDrive\Documents\ROYAL HOLLOWAY\data analysis\A2\boston2.csv')

l_o_t = df.to_records(index=False)
boston_data = list(l_o_t)

### PART A 
#split into train and test dataset
nrow_boston = len(boston_data) 
train_index = sample(range(nrow_boston), int(nrow_boston / 2) )

#get x_train and y_train
boston_train = []
for i in range(len(train_index)):
    boston_train.append(boston_data[i])


boston_test = []
for i in range(nrow_boston):
    if i not in train_index:
        boston_test.append(boston_data[i])

BTS_indx = []
no_samples = len(train_index)

for i in range(100):
    chosen_samples = choices(train_index, k = no_samples) # sample with replacemant 
    BTS_indx.append(chosen_samples) #Store the indices this will be useful when estimating train and test MSE 
    
### PART B     
#Now fit create 100 trees using BTS with randomly sampled attritubes at each node
BTS_trees = []
for i in range(100):
    BTS_trees.append(randomforest(boston_data, BTS_indx[i], 3))

print('Trees created!')
#%%
### PART C
#Now compute train MSE and test MSE
import numpy as np

#First compute predicted lables for train set and test set
train_yhat = np.zeros(len(boston_train))
test_yhat = np.zeros(len(boston_train))
y_train = np.zeros(len(boston_train)) # Also the train and test labels are obtained
y_test = np.zeros(len(boston_train))



for i in range(len(boston_train)): #First find the prediction made each tree and then find the average prediction
    tree_train_yhat = np.zeros(100) #This is done fpr evry sample in test set and train set
    tree_test_yhat = np.zeros(100) 
    
    for j in range(100):
        tree_train_yhat = BTS_trees[j].predict(boston_train[i])
        tree_test_yhat = BTS_trees[j]. predict(boston_test[i])
    
    train_yhat[i] = np.mean(tree_train_yhat)
    test_yhat[i] = np.mean(tree_test_yhat)
    
    y_train[i] = boston_train[i][-1] #Get the true label for each sample
    y_test[i] = boston_test[i][-1]

# Now compute the mse
n = len(boston_train)
Train_MSE = np.sum(np.square(train_yhat - y_train)) / n 
Test_MSE = np.sum(np.square(test_yhat - y_test)) / n 

print('The train MSE is:', Train_MSE)
print('The test MSE is:', Test_MSE)

### PART D
##define function which calculates train and test MSE for different values of B and h
#%%

def predict_randomforest(train_index, train, test, B, h):
    BTS_indx = []
    no_samples = len(train_index)

    for i in range(B):
        chosen_samples = choices(train_index, k = no_samples) # sample with replacemant 
        BTS_indx.append(chosen_samples) #Store the indices this will be useful when estimating train and test MSE 
    
    
    BTS_trees = []
    
    for i in range(B):
        BTS_trees.append(randomforest(boston_data, BTS_indx[i], h))
    
    #First compute predicted lables for train set and test set
    train_yhat = np.zeros(len(train))
    test_yhat = np.zeros(len(test))
    y_train = np.zeros(len(train)) # Also the train and test labels are obtained
    y_test = np.zeros(len(test))



    for i in range(len(train)): #First find the prediction made each tree and then find the average prediction
        tree_train_yhat = np.zeros(100) #This is done fpr evry sample in test set and train set
        tree_test_yhat = np.zeros(100) 
        
        for j in range(B):
            tree_train_yhat = BTS_trees[j].predict(train[i])
            tree_test_yhat = BTS_trees[j]. predict(test[i])
        
        train_yhat[i] = np.mean(tree_train_yhat)
        test_yhat[i] = np.mean(tree_test_yhat)
        
        y_train[i] = train[i][-1] #Get the true label for each sample
        y_test[i] = test[i][-1]

    # Now compute the mse
    n = len(boston_train)
    Train_MSE = np.sum(np.square(train_yhat - y_train)) / n 
    Test_MSE = np.sum(np.square(test_yhat - y_test)) / n
    
    return Train_MSE, Test_MSE


# Now compute Train and test MSE for B = 1, 5, 10, 25, 100, 1000 and h = 1,2,3,4,5,6,7,8

B = [1, 5, 10, 25, 100, 1000]
h = [1, 2, 3, 4, 5, 6, 7, 8]
Train_MSEs = np.zeros((8,6))
Test_MSEs = np.zeros((8,6))

for i in range(len(B)):
    for j in range(len(h)):
        Train_MSEs[j, i], Test_MSEs[j, i] = predict_randomforest(train_index, boston_train, boston_test, B[i], h[j]) 

    
print(Train_MSEs)
print(Test_MSEs)    

#%%
import matplotlib.pyplot as plt

plt.subplot(2,3,1).set_title('B = 1')
plt.plot(h, Train_MSEs[:,0])
plt.plot(h, Test_MSEs[:,0])
plt.ylabel('MSE')

plt.subplot(2, 3, 2).set_title('B = 5')
plt.plot(h, Train_MSEs[:,1])
plt.plot(h, Test_MSEs[:,1])

plt.subplot(2, 3, 3).set_title('B = 10') 
plt.plot(h, Train_MSEs[:,2])
plt.plot(h, Test_MSEs[:,2])

plt.subplot(2, 3, 4).set_title('B = 25')
plt.plot(h, Train_MSEs[:,3])
plt.plot(h, Test_MSEs[:,3])
plt.ylabel('MSE')
plt.xlabel('h')

plt.subplot(2, 3, 5).set_title('B = 100')
plt.plot(h, Train_MSEs[:,4])
plt.plot(h, Test_MSEs[:,4])
plt.xlabel('h')

plt.subplot(2, 3, 6).set_title('B = 1000')
plt.plot(h, Train_MSEs[:,5])
plt.plot(h, Test_MSEs[:,5])
plt.xlabel('h')

plt.subplots_adjust(wspace = 0.4, hspace = 0.6 ) # Adjust whitespaces so all titles  can be seen

#%%
#There seems to be a lot of random noise making it difficult to see the relationship between MSE and h.
#I will now plot average error rate for a given h to see if the relationship is easier to see

avg_trainMSE_h = np.zeros(8)
avg_testMSE_h = np.zeros(8)

for i in range(8):
    avg_trainMSE_h[i] = np.mean(Train_MSEs[i, :])
    avg_testMSE_h[i] = np.mean(Test_MSEs[i, :])

plt.plot(h, avg_trainMSE_h)
plt.plot(h, avg_testMSE_h)
plt.title('Average MSE agansit h')
plt.ylabel('Average MSE')
plt.xlabel('h')
plt.legend(['MSE train', 'MSE test'])


#%%
##
#Now i will explore the relasionship between B and MSE 
#I belive the best way to do this is too plot average test and train MSE agansit B.
#As seen in the previous plots there is high varibaility in large h. So, to better understand the relasionship between
#B and h i will only take the mean of h = 1-6
#I will plot the standar too to see how that chnages with b

avg_trainMSE_b = np.zeros(6)
avg_testMSE_b = np.zeros(6)
train_sd_b = np.zeros(6)
test_sd_b = np.zeros(6)

for i in range(6):
    avg_trainMSE_b[i] = np.mean(Train_MSEs[:6, i])
    avg_testMSE_b[i] = np.mean(Test_MSEs[:6, i])
    train_sd_b[i] = np.std(Train_MSEs[:6, i])
    test_sd_b[i] = np.std(Test_MSEs[:6, i])

plt.errorbar(np.log(B), avg_trainMSE_b, yerr=train_sd_b, fmt='o', color='darkblue',
             ecolor='slateblue', elinewidth=3, capsize=0)
plt.errorbar(np.log(B), avg_testMSE_b, yerr=test_sd_b, fmt='o', color='darkred',
             ecolor='lightcoral', elinewidth=3, capsize=0)
plt.title('Average MSE agnsit log(B)')
plt.xlabel('log(B)')
plt.ylabel('Average MSE')
plt.legend(['MSE train', 'MSE test'])

