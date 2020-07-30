# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:02:38 2020

@author: Sikander
"""

import degeneracy as dg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import Counter, defaultdict
import ctrnn
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import pandas as pd
import sys

targ0 = 2#sys.argv[1]
targ1 = 4

target_genotype_0 = np.load("{0}/target_genotype_{0}.npy".format(targ0))
target_genotype_1 = np.load("{0}/target_genotype_{0}.npy".format(targ1))

reps = 50
gens0 = len(np.load("{0}/avg_fitness_{0}_0.npy".format(targ0)))
gens1 = len(np.load("{0}/avg_fitness_{0}_0.npy".format(targ1)))
gs = len(np.load("{0}/best_individual_{0}_0.npy".format(targ0)))
af0 = np.zeros((reps,gens0))
bf0 = np.zeros((reps,gens0))
bi0 = np.zeros((reps,gs))
for i in range(reps):
    af0[i] = np.load("{0}/avg_fitness_{0}_{1}.npy".format(targ0, i)) 
    bf0[i] = np.load("{0}/best_fitness_{0}_{1}.npy".format(targ0, i))
    bi0[i] = np.load("{0}/best_individual_{0}_{1}.npy".format(targ0, i))
af1 = np.zeros((reps,gens1))
bf1 = np.zeros((reps,gens1))
bi1 = np.zeros((reps,gs))
for i in range(reps):
    af1[i] = np.load("{0}/avg_fitness_{0}_{1}.npy".format(targ1, i)) 
    bf1[i] = np.load("{0}/best_fitness_{0}_{1}.npy".format(targ1, i))
    bi1[i] = np.load("{0}/best_individual_{0}_{1}.npy".format(targ1, i))
    
dg.pca_compare(bi0, target_genotype_0, bi1, target_genotype_1, targ0, targ1)