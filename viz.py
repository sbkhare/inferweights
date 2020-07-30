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
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
import ctrnn
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import pandas as pd

nnsize = 30
stepsize = 0.01
start_index = int(95/stepsize)
WeightRange = 15

targ = 6
#targ0 = 2#sys.argv[1]
#targ1 = 4

target_genotype = np.load("N{1}/{0}/target_genotype_{0}.npy".format(targ, nnsize))
outputs = np.load("N{1}/{0}/target_output_{0}.npy".format(targ, nnsize))
#target_genotype_0 = np.load("N{1}/{0}/target_genotype_{0}.npy".format(targ0, nnsize))
#target_genotype_1 = np.load("N{1}/{0}/target_genotype_{0}.npy".format(targ1, nnsize))

ADJ_MAT = 0
PHASE_REDO = 0
P_C_A = 1
DOUBLE_PCA = 0



#if DOUBLE_PCA == 1:
#    reps = 50
#    gens0 = len(np.load("N{1}/{0}/avg_fitness_{0}_0.npy".format(targ0, nnsize)))
#    gens1 = len(np.load("N{1}/{0}/avg_fitness_{0}_0.npy".format(targ1, nnsize)))
#    gs = len(np.load("N{1}{0}/best_individual_{0}_0.npy".format(targ0, nnsize)))
#    af0 = np.zeros((reps,gens0))
#    bf0 = np.zeros((reps,gens0))
#    bi0 = np.zeros((reps,gs))
#    for i in range(reps):
#        af0[i] = np.load("N{2}/{0}/avg_fitness_{0}_{1}.npy".format(targ0, i, nnsize)) 
#        bf0[i] = np.load("N{2}/{0}/best_fitness_{0}_{1}.npy".format(targ0, i, nnsize))
#        bi0[i] = np.load("N{2}/{0}/best_individual_{0}_{1}.npy".format(targ0, i, nnsize))
#    af1 = np.zeros((reps,gens1))
#    bf1 = np.zeros((reps,gens1))
#    bi1 = np.zeros((reps,gs))
#    for i in range(reps):
#        af1[i] = np.load("N{2}/{0}/avg_fitness_{0}_{1}.npy".format(targ1, i, nnsize)) 
#        bf1[i] = np.load("N{2}/{0}/best_fitness_{0}_{1}.npy".format(targ1, i, nnsize))
#        bi1[i] = np.load("N{2}/{0}/best_individual_{0}_{1}.npy".format(targ1, i, nnsize))
#        
#    dg.pca_compare(bi0, target_genotype_0, bi1, target_genotype_1, targ0, targ1, nnsize)
    
if ADJ_MAT == 1:
    adjmat = np.zeros((nnsize,nnsize))
    k = 0
    for i in range(nnsize):
        for j in range(nnsize):
            adjmat[i][j] = target_genotype[k]*WeightRange
            k += 1
#    plt.figure()
    plt.matshow(adjmat, cmap='RdBu')
    plt.colorbar()

if PHASE_REDO == 1:
    for i in range(nnsize-2):
        if nnsize%2 == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel("Neuron {0}".format(i))
            ax.set_ylabel("Neuron {0}".format(i+1))
            ax.set_zlabel("Neuron {0}".format(i+2))
            ax.plot3D(outputs.T[i][start_index:],outputs.T[i+1][start_index:],outputs.T[i+2][start_index:])
            plt.savefig("N{0}/{2}/phase_{1}.png".format(nnsize, i, targ))
            
if P_C_A == 1:
    
    pca = PCA(n_components=None, svd_solver='full')
#    sc = StandardScaler()
#    std_outs = sc.fit_transform(outputs)
    res = pca.fit_transform(outputs) #pca.fit_transform(std_outs)
    df_PCA = pd.DataFrame(res[:, 0:9], columns=['1c', '2c', '3c', '4c', '5c', '6c', '7c', '8c', '9c'])
    s_Var = pd.Series(pca.explained_variance_ratio_, index=range(1, (res.shape[1] + 1)), name='explained_variance_ratio')
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(figsize=(12, 10), nrows=3, ncols=3)
    fig.suptitle("PCA on output for {0} neurons".format(nnsize))
    s_cumsum = s_Var.cumsum()
    n_eigen_95 = s_cumsum[(s_cumsum < 0.95)].shape[0]
    n = 9
    ind = np.arange(n)
    height = s_Var.iloc[:n].values
    width = 0.60
    xticklabels = (ind + 1)
    cmap = mpl.cm.get_cmap('hsv_r')
    norm = mpl.colors.Normalize(vmin=0, vmax=n)
    tab20 = cm.get_cmap('tab20').colors
    s_colors = tab20[0::2]
    s_edgecolors = tab20[1::2]
    ax1.bar(ind, height, width, color=s_colors, edgecolor=s_edgecolors, zorder=9, lw=1)
    ax1.set_xticks(ind)
    ax1.set_xticklabels(xticklabels)
    ax1.set_title('Explained variance ratio')
    ax1.annotate('95% with {:,d}\nsingular vectors'.format(n_eigen_95), xy=(0.97, 0.97), xycoords="axes fraction", ha='right', va='top')
    ax1.set_xlabel('Components')
    ax1.set_ylabel('%')
    ax1.grid()
    for dim, ax in zip(range(1, 10), [ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]):
        print('- Dim: {:d}'.format(dim))
        col = str(dim) + 'c'
        x = str(dim) + 'c'
        y = str(dim + 1) + 'c'
        xs = df_PCA[x].tolist()
        ys = df_PCA[y].tolist()
#        pca_colors = df_PCA.index.map(lambda x: '#ff9896' if x == '*' else '#7f7f7f') # dfPCA['color'].tolist()
#        pca_edgecolors = df_PCA.index.map(lambda x: '#ff9896' if x == '*' else '#7f7f7f')
        ax.plot(xs[start_index:], ys[start_index:])
        ax.plot(0,0, color='#2ca02c', marker='x', ms=16)
        ax.axhline(y=0, c='black', lw=0.75, ls='-.', zorder=2)
        ax.axvline(x=0, c='black', lw=0.75, ls='-.', zorder=2)
        ax.set_title('Components {dim1} and {dim2}'.format(dim1=dim, dim2=(dim + 1)) )
        ax.set_xlabel('Component {dim1:d}'.format(dim1=dim))
        ax.set_ylabel('Component {dim2:d}'.format(dim2=dim + 1))
        ax.grid()
        ax.axis('equal')
        ax.locator_params(axis='both', tight=True, nbins=6)
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.93, wspace=0.26, hspace=0.35)
    plt.savefig("N{1}/outputs_pca_{0}.png".format(targ, nnsize))
    df_PCA.to_csv("N{1}/outputs_pca_{0}.csv".format(targ, nnsize))
