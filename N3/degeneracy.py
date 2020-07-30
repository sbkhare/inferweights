# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:41:19 2020

@author: Sikander
"""

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

stepsize=0.01
fitness_threshold = 0.7
# NN Params
nnsize = 3
duration = 50
start_index = int(42/stepsize)
WeightRange = 15
BiasRange = 15
TimeConstMin = 1.0 #stepsize*10
TimeConstMax = 1.0 #2.0
###############################################################
targ = 14#sys.argv[1]


target_genotype = np.load("{0}/target_genotype_{0}.npy".format(targ))

reps = 50
gens = len(np.load("{0}/avg_fitness_{0}_0.npy".format(targ)))
gs = len(np.load("{0}/best_individual_{0}_0.npy".format(targ)))
af = np.zeros((reps,gens))
bf = np.zeros((reps,gens))
bi = np.zeros((reps,gs))
for i in range(reps):
    af[i] = np.load("{0}/avg_fitness_{0}_{1}.npy".format(targ, i)) 
    bf[i] = np.load("{0}/best_fitness_{0}_{1}.npy".format(targ, i))
    bi[i] = np.load("{0}/best_individual_{0}_{1}.npy".format(targ, i))

def eucdist(x,y):
    return np.sum((x-y)**2)**(1/2)

def filter_fit(params, fits, avgs, thresh):
    newpar = []
    newfit = []
    newavg = []
    for p, f, a in zip(params, fits, avgs):
        if f[-1] < thresh:
            continue
        else:
            newpar.append(p)
            newfit.append(f)
            newavg.append(a)
    newpar = np.stack(newpar)
    newfit = np.stack(newfit)
    newavg = np.stack(newavg)
    return newpar, newfit, newavg

#def dbscan(params, fitness, target_params, eps, min_samples):
#    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(params)
#    labels = clustering.labels_
#    counts = Counter(labels)
#    print(counts)
#    colors = [color for color in mcolors.BASE_COLORS] + [color for color in mcolors.TABLEAU_COLORS]
#    colors.remove('tab:cyan')
#    colors.remove('tab:olive')
#    colors.remove('tab:red')
#    num = 0
#    for lab in counts:
#        counts[lab] = colors[num]
#        num += 1
#    rl = []
#    num = 0
#    for par in params:
#        paramdist=eucdist(params[num], target_params)
#        rl.append([fitness[num], paramdist])
#        num += 1
#    rl = np.asarray(rl)
#    plt.figure()
#    num = 0
#    for lab in labels:
#        plt.plot(rl[num,0], rl[num,1], 'o', color=counts[lab], label=lab)
#        num += 1
#    plt.xlabel("Final fitness")
#    plt.ylabel("Param dist to target")
##    plt.legend()
#    plt.savefig("Paramdist_vs_fitness")
#    label_dict = {}
#    for para, lab in zip(params, labels):
#        if lab in label_dict:
#            label_dict[lab].append(para)
#        else:
#            label_dict[lab] = [para]
#    for lab in label_dict:
#        label_dict[lab] = np.stack(label_dict[lab], axis=0)
#
#        colnum = params.shape[1]
#        paramsl = np.concatenate(label_dict[lab])
#        data = pd.DataFrame([[paramsl[0], '0']], columns=["Value", "Gene"])
#        for idx, val in enumerate(paramsl):
#            data = data.append({"Value": val, "Gene": str(idx%colnum)}, ignore_index=True)
#        plt.figure()
#        sns.swarmplot(x="Gene", y="Value", data=data)
#        plt.title("Cluster {0}".format(lab))
#        plt.savefig("cluster_{0}_genotype.png".format(lab))
#        
#        
#        if nnsize == 2:
#            plt.figure()
#            plt.title("Phase portrait - Cluster {0}".format(lab))
#            plt.xlabel("Neuron 1")
#            plt.ylabel("Neuron 2")
#            time = np.arange(0.0,duration,stepsize)
#            for indiv in label_dict[lab]:
#                nn1 = ctrnn.CTRNN(nnsize)
#                nn1.setParameters(indiv,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
#                nn1.initializeState(np.zeros(nnsize))
#                outputs1 = np.zeros((len(time),nnsize))
#                step1 = 0
#                for t in time:
#                    nn1.step(stepsize)
#                    outputs1[step1] = nn1.Output
#                    step1 += 1
#                plt.plot(outputs1.T[0][start_index:],outputs1.T[1][start_index:])
#            nn = ctrnn.CTRNN(nnsize)
#            nn.setParameters(target_genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
#            nn.initializeState(np.zeros(nnsize))
#            outputs = np.zeros((len(time),nnsize))
#            step = 0
#            for t in time:
#                nn.step(stepsize)
#                outputs[step] = nn.Output
#                step += 1
#            plt.plot(outputs.T[0][start_index:],outputs.T[1][start_index:],'k', label="Target")
#            plt.savefig("cluster_{0}_phase.png".format(lab))
#        elif nnsize == 3:
#            fig = plt.figure()
#            ax = fig.add_subplot(111, projection='3d')
#            plt.title("Phase portrait - Cluster {0}".format(lab))
#            ax.set_xlabel("Neuron 1")
#            ax.set_ylabel("Neuron 2")
#            ax.set_zlabel("Neuron 3")
#            time = np.arange(0.0,duration,stepsize)
#            for indiv in label_dict[lab]:
#                nn1 = ctrnn.CTRNN(nnsize)
#                nn1.setParameters(indiv,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
#                nn1.initializeState(np.zeros(nnsize))
#                outputs1 = np.zeros((len(time),nnsize))
#                step1 = 0
#                for t in time:
#                    nn1.step(stepsize)
#                    outputs1[step1] = nn1.Output
#                    step1 += 1
#                ax.plot3D(outputs1.T[0][start_index:],outputs1.T[1][start_index:],outputs1.T[2][start_index:])
#            nn = ctrnn.CTRNN(nnsize)
#            nn.setParameters(target_genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
#            nn.initializeState(np.zeros(nnsize))
#            outputs = np.zeros((len(time),nnsize))
#            step = 0
#            for t in time:
#                nn.step(stepsize)
#                outputs[step] = nn.Output
#                step += 1
#            ax.plot3D(outputs.T[0][start_index:],outputs.T[1][start_index:],outputs.T[2][start_index:],'k', label="Target")
#            fig.savefig("cluster_{0}_phase.png".format(lab))
#            
#    return clustering #Find centroids, compare to target, phase portraits with each cluster colored with same color

def dendro_clusters(d):
    cluster_idxs = defaultdict(list)
    for c, pi in zip(d['color_list'], d['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))
    cluster_classes = {}
    for c, l in cluster_idxs.items():
        i_l = [d['ivl'][i] for i in l]
        cluster_classes[c] = i_l
    
    bestfit_params = {}
    avg_params = {}
    for cluster in cluster_classes:
        fits = {}
        params = np.zeros((len(cluster_classes[cluster]), gs))
        i = 0
        j = -1
        for idx in cluster_classes[cluster]:
            if idx == "*":
                j = i
                i += 1
            else:
                fits[idx] = bf[idx,-1]
                params[i] = bi[idx]
                i += 1
        if j != -1:
            params = np.delete(params, j, 0)
        bestfit_params[cluster] = (bi[max(fits, key=fits.get)], max(fits, key=fits.get)) #Parameters, index #
        avg_params[cluster] = np.mean(params, axis=0)
    
    time = np.arange(0.0,duration,stepsize)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    plt.title("Representative networks - best fit individuals")
    ax.set_xlabel("Neuron 1")
    ax.set_ylabel("Neuron 2")
    ax.set_zlabel("Neuron 3")
    nn = ctrnn.CTRNN(nnsize)
    nn.setParameters(target_genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
    nn.initializeState(np.zeros(nnsize))
    outputs = np.zeros((len(time),nnsize))
    step = 0
    for t in time:
        nn.step(stepsize)
        outputs[step] = nn.Output
        step += 1
    ax.plot3D(outputs.T[0][start_index:],outputs.T[1][start_index:],outputs.T[2][start_index:],'orange', label="Target", zorder=2)
    paramsl = []
    for cluster in bestfit_params:
        pars = bestfit_params[cluster][0]
        paramsl.append(pars)
        nn1 = ctrnn.CTRNN(nnsize)
        nn1.setParameters(pars,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
        nn1.initializeState(np.zeros(nnsize))
        outputs1 = np.zeros((len(time),nnsize))
        step1 = 0
        for t in time:
            nn1.step(stepsize)
            outputs1[step1] = nn1.Output
            step1 += 1
        ax.plot3D(outputs1.T[0][start_index:],outputs1.T[1][start_index:],outputs1.T[2][start_index:], zorder=1, color=cluster, label=str(bestfit_params[cluster][1])) #, label="No.{0}, fit={1}".format(bestfit_params[cluster][1], fits[bestfit_params[cluster][1]])
    paramsl = np.vstack(paramsl)
    plt.legend()
    colors = list(bestfit_params.keys())
    ax1 = fig.add_subplot(122)
    ax1.set_xlabel("Gene index #")
    ax1.set_ylabel("Value")
#    colnum = paramsl.shape[1]
#    paramsl = np.concatenate(paramsl)
#    data = pd.DataFrame([[paramsl[0], '0', colors[0]]], columns=["Value", "Gene", "Cluster"])
#    for idx, val in enumerate(paramsl):
#        if idx%colnum >= colnum-nnsize:
#            continue
#        else:
#            data = data.append({"Value": val, "Gene": str(idx%colnum)}, ignore_index=True)
    ax1.scatter(np.arange(nnsize**2 + nnsize), target_genotype[:-nnsize], color='orange', label="Target", zorder=2)
    for cluster in bestfit_params:
        ax1.scatter(np.arange(nnsize**2 + nnsize), bestfit_params[cluster][0][:-nnsize], color=cluster, zorder=1)
    plt.legend()
    fig1 = plt.figure()
    ax2 = fig1.add_subplot(121, projection='3d')
    plt.title("Representative networks - average individual")
    ax2.set_xlabel("Neuron 1")
    ax2.set_ylabel("Neuron 2")
    ax2.set_zlabel("Neuron 3")
    nn = ctrnn.CTRNN(nnsize)
    nn.setParameters(target_genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
    nn.initializeState(np.zeros(nnsize))
    outputs = np.zeros((len(time),nnsize))
    step = 0
    for t in time:
        nn.step(stepsize)
        outputs[step] = nn.Output
        step += 1
    ax2.plot3D(outputs.T[0][start_index:],outputs.T[1][start_index:],outputs.T[2][start_index:],'orange', label="Target", zorder=2)
    paramsl = []
    for cluster in avg_params:
        pars = avg_params[cluster]
        paramsl.append(pars)
        nn1 = ctrnn.CTRNN(nnsize)
        nn1.setParameters(pars,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
        nn1.initializeState(np.zeros(nnsize))
        outputs1 = np.zeros((len(time),nnsize))
        step1 = 0
        for t in time:
            nn1.step(stepsize)
            outputs1[step1] = nn1.Output
            step1 += 1
        ax2.plot3D(outputs1.T[0][start_index:],outputs1.T[1][start_index:],outputs1.T[2][start_index:], zorder=1, color=cluster)
    paramsl = np.vstack(paramsl)
    ax3 = fig1.add_subplot(122)
    ax3.set_xlabel("Gene index #")
    ax3.set_ylabel("Value")
    ax3.scatter(np.arange(nnsize**2 + nnsize), target_genotype[:-nnsize], color='orange', label="Target", zorder=2)
    for cluster in avg_params:
        ax3.scatter(np.arange(nnsize**2 + nnsize), avg_params[cluster][:-nnsize], color=cluster, zorder=1)
    return cluster_classes
    
def dendro(params, target_params, typ, dist_thresh):
    param2 = np.zeros((params.shape[0]+1,gs))
    param2[:params.shape[0]] = params
    param2[-1] = target_params
    param3 = param2[:,:-nnsize]
    Z = linkage(param3, typ)
    plt.figure()
    d = dendrogram(Z, orientation='left', labels=list(np.arange(params.shape[0]))+['*'], color_threshold=dist_thresh*max(Z[:,2]))
    plt.title("{0}-neuron results: dendrogram - {1}".format(nnsize, typ))
    plt.savefig("dendrogram_{0}.png".format(targ))
    
    colors_seen = set()
    time = np.arange(0.0,duration,stepsize)
    clusters = {}
    clust_list = dendro_clusters(d) #this is actually a dict
    print(clust_list)
    for color in clust_list:
        for num in clust_list[color]:
            if num == '*':
                continue
            else:
                if color not in colors_seen:
                    colors_seen.add(color)
                    clusters[color] = [params[num,:]]
                    print(color)
                    print(num)
                else:
                    clusters[color].append(params[num,:])
                    print(num)
    for color in clusters:
        clusters[color] = np.stack(clusters[color], axis=0)
        colnum = params.shape[1]
        paramsl = np.concatenate(clusters[color])
        data = pd.DataFrame([[paramsl[0], '0']], columns=["Value", "Gene"])
        for idx, val in enumerate(paramsl):
            if idx%colnum >= colnum-nnsize:
                continue
            else:
                data = data.append({"Value": val, "Gene": str(idx%colnum)}, ignore_index=True)
        plt.figure()
        sns.swarmplot(x="Gene", y="Value", data=data)
        plt.title("Cluster {0}".format(color))
        plt.savefig("{0}_cluster_{1}_genotype.png".format(targ, color))
        if nnsize == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plt.title("Phase portrait - Cluster {0}".format(color))
            ax.set_xlabel("Neuron 1")
            ax.set_ylabel("Neuron 2")
            ax.set_zlabel("Neuron 3")
            nn = ctrnn.CTRNN(nnsize)
            nn.setParameters(target_genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
            nn.initializeState(np.zeros(nnsize))
            outputs = np.zeros((len(time),nnsize))
            step = 0
            for t in time:
                nn.step(stepsize)
                outputs[step] = nn.Output
                step += 1
            ax.plot3D(outputs.T[0][start_index:],outputs.T[1][start_index:],outputs.T[2][start_index:],'k', label="Target", zorder=2)
            for pars in clusters[color]:
                nn1 = ctrnn.CTRNN(nnsize)
                nn1.setParameters(pars,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
                nn1.initializeState(np.zeros(nnsize))
                outputs1 = np.zeros((len(time),nnsize))
                step1 = 0
                for t in time:
                    nn1.step(stepsize)
                    outputs1[step1] = nn1.Output
                    step1 += 1
                ax.plot3D(outputs1.T[0][start_index:],outputs1.T[1][start_index:],outputs1.T[2][start_index:], zorder=1)
            plt.savefig("{0}_cluster_{1}_phase.png".format(targ, color))
    return d, clusters, clust_list

def princ_comp(params, target_params):
    param2 = np.zeros((params.shape[0]+1,gs))
    param2[:params.shape[0]] = params
    param2[-1] = target_params
    param3 = param2[:,:-nnsize]
    indices = list(np.arange(params.shape[0]))+['*']
    pca = PCA(n_components=None, svd_solver='full')
    res = pca.fit_transform(param3)
    df_PCA = pd.DataFrame(res[:, 0:9], columns=['1c', '2c', '3c', '4c', '5c', '6c', '7c', '8c', '9c'], index=indices)
    s_Var = pd.Series(pca.explained_variance_ratio_, index=range(1, (res.shape[1] + 1)), name='explained_variance_ratio')
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(figsize=(12, 10), nrows=3, ncols=3)
    fig.suptitle("PCA on results for {0} neurons".format(nnsize))
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
        pca_colors = df_PCA.index.map(lambda x: '#ff9896' if x == '*' else '#7f7f7f') # dfPCA['color'].tolist()
        pca_edgecolors = df_PCA.index.map(lambda x: '#ff9896' if x == '*' else '#7f7f7f')
        ax.scatter(xs, ys, c=pca_colors, marker='o', edgecolor=pca_edgecolors, lw=0.5, s=30, zorder=5, rasterized=True)
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
    plt.savefig("n{0}_pca_{1}.png".format(nnsize, targ))
    df_PCA.to_csv("n{0}_pca_{1}.csv".format(nnsize, targ))
    
def pca_compare(params0, target0, params1, target1, num0, num1):
    param2 = np.zeros((params0.shape[0]+params1.shape[0]+2, params1.shape[1]))
    param2[:params0.shape[0]] = params0
    param2[-1] = target0
    param2[params0.shape[0]+1:-1] = params1
    param2[-1] = target1
    param3 = param2[:,:-nnsize]
    indices = list(np.arange(params0.shape[0])) + ['*1'] + list(params0.shape[0]+np.arange(params1.shape[0])) + ['*2']
    pca = PCA(n_components=None, svd_solver='full')
    res = pca.fit_transform(param3)
    df_PCA = pd.DataFrame(res[:, 0:9], columns=['1c', '2c', '3c', '4c', '5c', '6c', '7c', '8c', '9c'], index=indices)
    s_Var = pd.Series(pca.explained_variance_ratio_, index=range(1, (res.shape[1] + 1)), name='explained_variance_ratio')
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(figsize=(12, 10), nrows=3, ncols=3)
    fig.suptitle("PCA on results for {0} neurons".format(nnsize))
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
        pca_colors = df_PCA.index.map(lambda x: 'salmon' if x == '*1' else ('gold' if x == '*2' else ('royalblue' if int(x)<params0.shape[0] else 'limegreen')))#7f7f7f') # dfPCA['color'].tolist()
        pca_edgecolors = df_PCA.index.map(lambda x: 'darkred' if x == '*1' else ('darkorange' if x == '*2' else ('navy' if int(x)<params0.shape[0] else 'darkgreen')))
        ax.scatter(xs, ys, c=pca_colors, marker='o', edgecolor=pca_edgecolors, lw=0.5, s=30, zorder=5, rasterized=True)
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
    plt.savefig("n{0}_double_pca_{1}_{2}.png".format(nnsize, num0, num1))
    df_PCA.to_csv("n{0}_double_pca_{1}_{2}.csv".format(nnsize, num0, num1))

if __name__=='__main__':
#    clust = dbscan(bi, bf[:,-1], target_genotype, 1, 3)    

    thr = np.mean(bf[:,-1])
    bi1, bf1, af1 = filter_fit(bi, bf, af, thr)
    
#    x_embedded = TSNE(n_components=2, perplexity=5, learning_rate=200).fit_transform(bi)
#    plt.figure()
#    plt.scatter(x_embedded[:,0], x_embedded[:,1], color='g')
#    plt.title("t-SNE visualization - {0} neurons".format(nnsize))
#    plt.savefig("n{0}_tsne_{1}.png".format(nnsize, targ))
    
#    princ_comp(bi, target_genotype)
    
#    d1, c1, cl1 = dendro(bi1, target_genotype, 'weighted', 0.6)
#    plt.figure()
#    plt.title("3 neurons")
#    plt.xlabel("Fitness")
#    plt.ylabel("Parameter distance")
#    x = []
#    y = []
#    for clust in c1:
#        distlist = []
#        fitlist = []
#        pt_label = []
#        i = 0
#        for gene in c1[clust]:
#            bf_idx = cl1[clust][i]
#            i += 1
#            if bf_idx == '*':
#                continue
#            else:
#                fitlist.append(bf1[bf_idx,-1])
#                pt_label.append(bf_idx)
#                x.append(bf1[bf_idx,-1])
#                dist = eucdist(target_genotype, gene)
#                distlist.append(dist)
#                y.append(dist)
#        plt.scatter(fitlist, distlist, c=clust)
#        for j, txt in enumerate(pt_label):
#            plt.annotate(txt, (fitlist[j], distlist[j]))
#    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), c='k', linestyle='dashed') #ax+b; coeff = np.polyfit(x,y,1), a=coeff[0], b=coeff[1]
#    plt.savefig("fit_vs_dist_{0}.png".format(targ))
            
#    plt.figure()
#    plt.title("Average fitness")
#    plt.xlabel("Generations")
#    for ft in af:
#        plt.plot(np.arange(1, len(ft)+1), ft)
#    plt.savefig("fitness_avg.png")
#    plt.figure()
#    plt.title("Best fitness")
#    plt.xlabel("Generations")
#    for ft in bf:
#        plt.plot(np.arange(1, len(ft)+1), ft)
#    plt.axhline(y=thr, color='r', linestyle='dashed')
#    plt.savefig("fitness_best.png")
#   
#    clust_arr = np.zeros(bi1.shape[0], dtype=str)
#    for clust in cl1:
#        for ind in cl1[clust]:
#            if ind == '*':
#                continue
#            else:
#                clust_arr[ind] = clust
#    colnum = bi1.shape[1]
#    bi_cat = np.concatenate(bi1)
#    data = pd.DataFrame([[15*target_genotype[0], '0', "Target"]], columns=["Value", "Gene", "Cluster"])
#    for idx, val in enumerate(target_genotype):
#        if idx == 0:
#            continue
#        else:
#            if idx%colnum >= colnum-nnsize:
##                data = data.append({"Value": val, "Gene": str(idx%colnum), "Type": "Target"}, ignore_index=True)
#                continue
#            else:
#                data = data.append({"Value": 15*val, "Gene": str(idx%colnum), "Cluster": "Target"}, ignore_index=True)
#    for idx, val in enumerate(bi_cat):
#        if idx%colnum >= colnum-nnsize:
##            data = data.append({"Value": 15*val, "Gene": str(idx%colnum), "Type": "Inferred"}, ignore_index=True)
#            continue
#        else:
#            data = data.append({"Value": 15*val, "Gene": str(idx%colnum), "Cluster": clust_arr[idx//colnum]}, ignore_index=True)
#    plt.figure()
##    plt.title("Parameter Values", fontsize=14)
#    g = sns.swarmplot(x="Gene", y="Value", hue="Cluster", data=data) #, palette="Set2"
#    plt.ylabel("Parameter Value", fontsize=18)
#    plt.xlabel("Parameter Index", fontsize=18)
#    plt.title("3 neurons")
#    plt.xticks(fontsize=14)
#    plt.yticks(fontsize=14)
#    plt.tight_layout()
#    plt.savefig("params_{0}".format(targ))
    
    estimated_pars = np.mean(bi1, axis=0)
    plt.figure()
    plt.scatter(np.arange(bi1.shape[1]-nnsize), 15*target_genotype[:-nnsize], color='r', label='Target')
    plt.scatter(np.arange(bi1.shape[1]-nnsize), 15*estimated_pars[:-nnsize], color='g', label='Estimated')
    plt.legend()
    plt.ylabel("Parameter Value", fontsize=14)
    plt.xlabel("Parameter Index", fontsize=14)
    plt.title("Estimated parameter values (average)", fontsize=18)
    plt.savefig("n{0}_avgestimate_{1}.png".format(nnsize, targ))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Phase portrait")
    ax.set_xlabel("Neuron 1")
    ax.set_ylabel("Neuron 2")
    ax.set_zlabel("Neuron 3")
    nn_1 = ctrnn.CTRNN(nnsize)
    nn_1.setParameters(target_genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
    nn_1.initializeState(np.zeros(nnsize))
    time = np.arange(0.0,duration,stepsize)
    outputs_targ = np.zeros((len(time),nnsize))
    step = 0
    for t in time:
        nn_1.step(stepsize)
        outputs_targ[step] = nn_1.Output
        step += 1
    nn_2 = ctrnn.CTRNN(nnsize)
    nn_2.setParameters(estimated_pars,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
    nn_2.initializeState(np.zeros(nnsize))
    time = np.arange(0.0,duration,stepsize)
    outputs_est = np.zeros((len(time),nnsize))
    step = 0
    for t in time:
        nn_2.step(stepsize)
        outputs_est[step] = nn_2.Output
        step += 1
    ax.plot3D(outputs_targ.T[0][start_index:],outputs_targ.T[1][start_index:],outputs_targ.T[2][start_index:],'r', label="Target",zorder=1)
    ax.plot3D(outputs_est.T[0][start_index:],outputs_est.T[1][start_index:],outputs_est.T[2][start_index:],'g', label="Estimated",zorder=2)
    ax.legend()
    plt.savefig("n{0}_avgestimate_phase_{1}.png".format(nnsize, targ))