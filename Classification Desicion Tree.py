# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 22:08:40 2019

@author: Rawan Abdulelsadig - 35324987
"""

import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from scipy import stats
from sklearn.tree import DecisionTreeClassifier as sklearnDT
from sklearn.model_selection import StratifiedKFold


class TreeNode:
    def __init__(self , y = None , n_samples=None, left = None , right = None):
        self._y   = y             # a variable that contains the split threshold
        self._n_samples = n_samples
        self._left = left             # a pointer to the left tree or leaf
        self._right = right            # a pointer to the right tree or leaf
        
class LeafNode:
    def __init__(self , n_samples= None , ModeClass = None):
        self._n_samples = n_samples  # the number of samples in the node
        self._ModeClass = ModeClass  # the assigned class of the node
        
        
class DecisionTree:
    
    def __init__(self, min_leaf_samples = 1, min_leaf_split = 2 , max_depth = 50, criterion = 'gini',
                 infogain_threshold = 0.0, random_split = False, random_state = None):

        if min_leaf_samples <= 0 or min_leaf_split <= 0 or max_depth <= 0 :
            raise Exception('min_leaf_samples, min_leaf_split and max_depth should be a positive non-zero integers.')
        
        if type(min_leaf_samples) != int or type(min_leaf_split) != int or type(max_depth) != int :
            raise Exception('min_leaf_samples, min_leaf_split and max_depth should be integers.')
            
        if criterion != 'gini' and criterion != 'entropy':
            raise Exception('Criterion must be "gini" or "entropy".')
            
        if min_leaf_split < 2* min_leaf_samples:
            raise Exception('min_leaf_split should be greater than or equal to 2*min_leaf_samples.')
            
        self._min_leaf_samples = min_leaf_samples
        self._min_leaf_split = min_leaf_split
        self._max_depth = max_depth
        self._criterion = criterion
        self._infogain_threshold = infogain_threshold
        self._random_split = random_split
        self._random_seed = random_state
        self._root = None
        self._tree_depth = 0
        
        
    def _Purity(self , D1 , D2 , D, classes):
        if self._criterion == 'gini':
            p_l = 0.0
            p_r = 0.0
            for i in classes:
                pl = len(D1[D1[:,-1] == i])/ len(D1)# if len(D1) != 0 else 0
                pr = len(D2[D2[:,-1] == i])/ len(D2)# if len(D2) != 0 else 0
                p_l += pl**2
                p_r += pr**2
            purity = (len(D1)/len(D))*(1-p_l) + (len(D2)/len(D))*(1-p_r)
            del pl , pr , p_l , p_r
            
        if self._criterion == 'entropy':
            p_l = 0.0
            p_r = 0.0
            for i in classes:
                pl = len(D1[D1[:,-1] == i])/ len(D1)# if len(D1) != 0 else 0
                pr = len(D2[D2[:,-1] == i])/ len(D2)# if len(D2) != 0 else 0
                p_l -= pl*np.log2(pl if pl != 0 else 1e-20)
                p_r -= pr*np.log2(pr if pr != 0 else 1e-20)
            purity = (len(D1)/len(D)) * p_l + (len(D2)/len(D)) * p_r
            del pl , pr , p_l , p_r
            
        return purity
    
    def _information_gain(self, D1 , D2 , D):
        classes = np.unique(D[:,-1])
        p_l = 0.0
        p_r = 0.0
        p_d = 0.0
        for i in classes:
            pl = len(D1[D1[:,-1] == i])/ len(D1)# if len(D1) != 0 else 0
            pr = len(D2[D2[:,-1] == i])/ len(D2)# if len(D2) != 0 else 0
            pd  = len(D[D[:,-1] == i])/ len(D)   # if len(D) != 0 else 0
            p_l -= pl*np.log2(pl if pl != 0 else 1e-20)
            p_r -= pr*np.log2(pr if pr != 0 else 1e-20)
            p_d -= pd*np.log2(pd if pd != 0 else 1e-20)
        infogain = p_d -((len(D1)/len(D)) * p_l + (len(D2)/len(D)) * p_r)
        del pl , pr , pd , p_l , p_r , p_d
        return infogain
    
    def _Split(self , D):
        best_purity = np.float('inf')
        best_split = None
        best_D1 = D
        best_D2 = D
        classes = np.unique(D[:,-1])
        columns = range(D.shape[1] - 1)
        iterations = 0
        np.random.seed(self._random_seed)
        while(best_purity == np.float('inf')) and (iterations < 1000):
        # when using random splits, a proper split may not be found (best_purity may not get updated)
        # so a new set of midpoints will be needed, this is why thi while loop is needed.
            iterations += 1
            cols = np.copy(columns)
            if self._random_split: cols = np.random.choice(columns, size=1)
            for col in cols:
                ordered = D[D[:,col].argsort()] #  a sorted version of the continous variables
                ordered = np.unique(ordered, axis=0)
                midpoints = ((ordered[1:,col] + ordered[:-1,col]) / 2)
                if self._random_split:
                    midpoints = np.sort(np.random.choice(midpoints, size=int(np.ceil(0.25*len(midpoints))), replace=False))
                for s in midpoints:
                    D1 = D[D[:,col] <= s] # the left subset of the data
                    D2 = D[D[:,col] >  s] # the right subset of the data
                    if (len(D1) >= self._min_leaf_samples) and (len(D2) >= self._min_leaf_samples):
                        purity = self._Purity(D1 , D2 , D, classes)
                        
                        if purity < best_purity : # if the split at y produced a lower purity value:
                            best_purity = purity
                            #print(best_purity)
                            best_split = (col , s)
                            best_D1 = D1
                            best_D2 = D2                            
                    if best_purity == 0.0 : break
                if best_purity == 0.0 : break         # Stopping all loops once a pure split is found
            if best_purity == 0.0 : break 
        return best_split, best_purity, best_D1, best_D2
    
    
    def _buildTree(self , data , min_split, level): # A recursive method that keeps splitting the data untill it reaches a leaf
        if len(data) >= min_split : split , _ , D1 , D2 = self._Split(data)
        # when the size of data is less than the minimum size for a split, return it as a leaf node:
        else : return LeafNode(len(data), int(stats.mode(data[:,-1]).mode[0]))
        
        # if there is no significant information gain, just return the parent as a leaf
        if self._information_gain(D1 , D2 , data) <= self._infogain_threshold:
            return LeafNode(len(data), int(stats.mode(data[:,-1]).mode[0]))
        else:
            # recursively build the sub-trees as long as the maximum depth is not reached:
            if level < self._max_depth-1:
                left_tree = self._buildTree(D1, min_split, level+1)
                right_tree = self._buildTree(D2 , min_split, level+1)
            else:
                left_tree = LeafNode(len(D1), int(stats.mode(D1[:,-1]).mode[0])) 
                right_tree = LeafNode(len(D2), int(stats.mode(D2[:,-1]).mode[0]))

            return TreeNode(y = split , n_samples= len(data), left = left_tree , right = right_tree)
    
    
    def fit(self , X , y ):
        #import pdb; pdb.set_trace()
        self._tree_depth = 0 # to resit if previously fitted
        data = np.append(X , y.reshape(y.shape[0], 1) , axis = 1) # adjusting to the acceptable form of the data
        self._root = self._buildTree(data, min_split = self._min_leaf_split, level = 0)
        return
    
    
    def _traversePredict(self , node, X, preds , slic):
        if type(node) is TreeNode :
            i , thrsh = node._y
            # Extracting boolean slices of the data based on the condition
            sl = X[:,i] <= thrsh
            sr = X[:,i] > thrsh
            # Recursively predicting each slice
            # The boolean slices are updated by taking the intersection in each level
            if sum(sl) != 0 : self._traversePredict(node._left ,X , preds , np.logical_and(slic, sl))
            if sum(sr) != 0 : self._traversePredict(node._right,X , preds , np.logical_and(slic, sr))
            # the sum of a slice is the number of trues in it
            
        elif type(node) is LeafNode : # When reaching the leaf node
            preds[slic] = node._ModeClass # the mode class of the leaf node is the predicted class
        return 
    
        
    def predict(self , X):
        #import pdb; pdb.set_trace()
        predictions = np.ones((X.shape[0])) * -1 # Initializing an emtpty array filled with -1
        slic = (predictions == -1) # A boolean slice that includes the whole array as a start
        self._traversePredict(self._root, X, predictions , slic)
        return predictions
        
    def _traverseTree(self , node, level = 0, tabs='', print_levels = False):
        if self._tree_depth <= level : self._tree_depth = level
        if type(node) is TreeNode :
            if print_levels: print(tabs+'Level '+str(level)+': Number of Samples in the Node :', node._n_samples)
            if print_levels: print(tabs+'Level '+str(level)+': Splitting by: X'+str(node._y[0])+' <= '+str(node._y[1]))
            if print_levels: print(tabs+'Level '+str(level)+': Going left:')
            self._traverseTree(node._left , level +1 , tabs+'\t', print_levels)
            if print_levels: print(tabs+'Level '+str(level)+': Going right:')
            self._traverseTree(node._right , level +1,  tabs+'\t', print_levels)
        elif type(node) is LeafNode :
            if print_levels: print(tabs+'Level '+str(level)+': Number of samples in the leaf: ', str(node._n_samples), ' of Class: '+ str(node._ModeClass))
        return
    
    def viewTree(self):
        self._traverseTree(self._root , print_levels = True)
        return
    
    def get_treeDepth(self):
        if self._tree_depth == 0 :
            self._traverseTree(self._root,print_levels = False)
        return self._tree_depth
    
#%%
# Importing the data: Breast Cancer Wisconsin (Diagnostic) Data Set
column_names = ['Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion',
                'Single Epithelial Cell Size','Bare Nuclei' , 'Bland Chromatin', 'Normal Nucleoli',
                'Mitoses','Class']
data = pd.read_csv('C:/Users/D/OneDrive - Lancaster University/Notebooks/breast-cancer-wisconsin.data', names = column_names)

# First, handling the missing data in column 'Bare Nuclei' by replacing them with the mode:
mode = data.loc[:,'Bare Nuclei'].mode()[0]
data.loc[:,'Bare Nuclei'] = data.loc[:,'Bare Nuclei'].replace('?' , mode).astype('int64')

# Chaning the data to the form that is accepted by the desicion tree: X and y numpy arrays
y = np.array(data.Class)
X = np.array(data.drop("Class", axis=1))

#%%
# Cross Validation and statistics gathering:

my_accuracy = []
sk_accuracy = []
my_F1Score = []
sk_F1Score = []
my_precision = []
sk_precision = []
my_recall = []
sk_recall = []
my_fitting_time = []
sk_fitting_time = []
my_prediction_time = []
sk_prediction_time = []
my_Model_size = []
sk_Model_size = []
my_tree_depth = []
sk_tree_depth = []

min_leaf_splits = [i for i in range(2, int(len(X)*0.25)+1)]
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

for min_leaf_split in min_leaf_splits:
    
    my_accuracy_ = []
    sk_accuracy_ = []
    my_F1Score_ = []
    sk_F1Score_ = []
    my_precision_ = []
    sk_precision_ = []
    my_recall_ = []
    sk_recall_ = []
    my_fitting_time_ = []
    sk_fitting_time_ = []
    my_prediction_time_ = []
    sk_prediction_time_ = []
    my_Model_size_ = []
    sk_Model_size_ = []
    my_tree_depth_ = []
    sk_tree_depth_ = []
    
    for train_index, valid_index in cv.split(X, y):
        X_train , y_train = X[train_index] , y[train_index]
        X_valid , y_valid = X[valid_index] , y[valid_index]
        
        DT = DecisionTree(min_leaf_split=min_leaf_split, criterion='entropy', random_split=False) # Changes are made here
        skDT = sklearnDT(min_samples_split=min_leaf_split, criterion='entropy', splitter='best')  # And here
        
        # Fitting and Measuring time
        startt = time.perf_counter()
        DT.fit(X_train , y_train);
        endt = time.perf_counter()
        my_fitting_time_.append(endt - startt)
        
        startt = time.perf_counter()
        skDT.fit(X_train , y_train);
        endt = time.perf_counter()
        sk_fitting_time_.append(endt - startt)
        
        # Model Size
        my_Model_size_.append(sys.getsizeof(pickle.dumps(DT)))
        sk_Model_size_.append(sys.getsizeof(pickle.dumps(skDT)))
        
        # Tree Depth
        my_tree_depth_.append(DT.get_treeDepth())
        sk_tree_depth_.append(skDT.get_depth())
        
        # Predicting and Measuring time
        startt = time.perf_counter()
        DT_preds = DT.predict(X_valid) 
        endt = time.perf_counter()
        my_prediction_time_.append(endt - startt)
        
        startt = time.perf_counter()
        skDT_preds = skDT.predict(X_valid)
        endt = time.perf_counter()
        sk_prediction_time_.append(endt - startt)
        
        # Prediction Metrics
        my_accuracy_.append(metrics.accuracy_score(y_valid , DT_preds))
        sk_accuracy_.append(metrics.accuracy_score(y_valid , skDT_preds))
        my_F1Score_.append(metrics.f1_score(y_valid , DT_preds, average='weighted'))
        sk_F1Score_.append(metrics.f1_score(y_valid , skDT_preds, average='weighted'))
        my_precision_.append(metrics.precision_score(y_valid , DT_preds , labels = [2,4], pos_label=4, average='binary'))
        sk_precision_.append(metrics.precision_score(y_valid , skDT_preds, labels = [2,4], pos_label=4, average='binary'))
        my_recall_.append(metrics.recall_score(y_valid , DT_preds , labels = [2,4], pos_label=4, average='binary'))
        sk_recall_.append(metrics.recall_score(y_valid , skDT_preds , labels = [2,4], pos_label=4, average='binary'))
    
    my_accuracy.append(np.mean(my_accuracy_))
    sk_accuracy.append(np.mean(sk_accuracy_))
    my_F1Score.append(np.mean(my_F1Score_))
    sk_F1Score.append(np.mean(sk_F1Score_))
    my_precision.append(np.mean(my_precision_))
    sk_precision.append(np.mean(sk_precision_))
    my_recall.append(np.mean(my_recall_))
    sk_recall.append(np.mean(sk_recall_))
    my_fitting_time.append(np.mean(my_fitting_time_))
    sk_fitting_time.append(np.mean(sk_fitting_time_))
    my_prediction_time.append(np.mean(my_prediction_time_))
    sk_prediction_time.append(np.mean(sk_prediction_time_))
    my_Model_size.append(np.mean(my_Model_size_))
    sk_Model_size.append(np.mean(sk_Model_size_))
    my_tree_depth.append(np.mean(my_tree_depth_))
    sk_tree_depth.append(np.mean(sk_tree_depth_))

Results = {'Min Leaf Split': min_leaf_splits,
           'Accuracy':my_accuracy , 'SKlearn Accuracy':sk_accuracy,
           'F1-Score':my_F1Score , 'SKlearn F1-Score':sk_F1Score,
           'Precision':my_precision, 'SKlearn Presicion':sk_precision,
           'Recall':my_recall, 'SKlearn Recall':sk_recall,
           'Fitting Time':my_fitting_time, 'SKlearn Fitting Time': sk_fitting_time,
           'Prediction Time': my_prediction_time, 'SKlearn Prediction Time':sk_prediction_time,
           'Tree Depth': my_tree_depth, 'SKlearn Tree Depth':sk_tree_depth,
           'Model Size': my_Model_size, 'SKlearn Model Size': sk_Model_size}

Results = pd.DataFrame(Results)
Results.to_csv('C:/Users/D/OneDrive - Lancaster University/Notebooks/DT-infogain0.0-best-entropy-Results.csv', index =False)
        
    