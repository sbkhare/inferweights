# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 08:52:39 2020

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
nnsize = 4
duration = 50
start_index = int(42/stepsize)
WeightRange = 15
BiasRange = 15
TimeConstMin = 1.0 #stepsize*10
TimeConstMax = 1.0 #2.0
###############################################################
targ = 4#sys.argv[1]

target_genotype = np.load("N{1}/{0}/target_genotype_{0}.npy".format(targ, nnsize))
target_output = np.load("N{1}/{0}/target_output_{0}.npy".format(targ, nnsize))


def eucdist(x,y):
    return np.sum((x-y)**2)**(1/2)

def load_results(target, size, reps):
    gens = len(np.load("N{1}/{0}/avg_fitness_{0}_2.npy".format(targ, nnsize)))
    gs = len(np.load("N{1}/{0}/best_individual_{0}_2.npy".format(targ, nnsize)))
    af = np.zeros((reps,gens))
    bf = np.zeros((reps,gens))
    bi = np.zeros((reps,gs))
    for i in range(1,reps): #range(3,reps) for N30/5
        if i not in [46]:
            af[i] = np.load("N{2}/{0}/avg_fitness_{0}_{1}.npy".format(targ, i, nnsize)) 
            bf[i] = np.load("N{2}/{0}/best_fitness_{0}_{1}.npy".format(targ, i, nnsize))
            bi[i] = np.load("N{2}/{0}/best_individual_{0}_{1}.npy".format(targ, i, nnsize))
    return gens, gs, af, bf, bi

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
#            if idx%colnum > colnum-nnsize:
#                continue
#            else:
#                data = data.append({"Value": val, "Gene": str(idx%colnum)}, ignore_index=True)
#        plt.figure()
#        sns.swarmplot(x="Gene", y="Value", data=data)
#        plt.title("Cluster {0}".format(lab))
#        plt.savefig("{0}_cluster_{1}_genotype.png".format(targ, lab))
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
#            plt.savefig("{0}_cluster_{1}_phase.png".format(targ, lab))
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
    if nnsize ==3:
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
    plt.savefig("N{1}/dendrogram_{0}.png".format(targ, nnsize))
    
    colors_seen = set()
    time = np.arange(0.0,duration,stepsize)
    clusters = {}
    clust_list = dendro_clusters(d)
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
        plt.savefig("N{2}/{0}_cluster_{1}_genotype.png".format(targ, color, nnsize))
        
        if nnsize == 2:
            plt.figure()
            plt.title("Phase portrait - Cluster {0}".format(color))
            plt.xlabel("Neuron 1")
            plt.ylabel("Neuron 2")
            nn = ctrnn.CTRNN(nnsize)
            nn.setParameters(target_genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
            nn.initializeState(np.zeros(nnsize))
            outputs = np.zeros((len(time),nnsize))
            step = 0
            for t in time:
                nn.step(stepsize)
                outputs[step] = nn.Output
                step += 1
            plt.plot(outputs.T[0][start_index:],outputs.T[1][start_index:], 'k', label="Target", zorder=2)
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
                plt.plot(outputs1.T[0][start_index:],outputs1.T[1][start_index:], zorder=1)
            plt.savefig("N{2}/{0}_cluster_{1}_phase.png".format(targ, color, nnsize))
        elif nnsize == 3:
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
            plt.savefig("N{2}/{0}_cluster_{1}_phase.png".format(targ, color, nnsize))
        elif nnsize == 4:
            fig = plt.figure()
            plt.suptitle("Phase portrait - Cluster {0}".format(color))
            ax1 = fig.add_subplot(221, projection='3d')
            ax1.set_xlabel("Neuron 1")
            ax1.set_ylabel("Neuron 2")
            ax1.set_zlabel("Neuron 3")
            nn = ctrnn.CTRNN(nnsize)
            nn.setParameters(target_genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
            nn.initializeState(np.zeros(nnsize))
            outputs = np.zeros((len(time),nnsize))
            step = 0
            for t in time:
                nn.step(stepsize)
                outputs[step] = nn.Output
                step += 1
            ax1.plot3D(outputs.T[0][start_index:],outputs.T[1][start_index:],outputs.T[2][start_index:],'k', label="Target", zorder=2)
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
                ax1.plot3D(outputs1.T[0][start_index:],outputs1.T[1][start_index:],outputs1.T[2][start_index:], zorder=1)
            ax2 = fig.add_subplot(222, projection='3d')
            ax2.set_xlabel("Neuron 1")
            ax2.set_ylabel("Neuron 2")
            ax2.set_zlabel("Neuron 4")
            nn = ctrnn.CTRNN(nnsize)
            nn.setParameters(target_genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
            nn.initializeState(np.zeros(nnsize))
            outputs = np.zeros((len(time),nnsize))
            step = 0
            for t in time:
                nn.step(stepsize)
                outputs[step] = nn.Output
                step += 1
            ax2.plot3D(outputs.T[0][start_index:],outputs.T[1][start_index:],outputs.T[3][start_index:],'k', label="Target", zorder=2)
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
                ax2.plot3D(outputs1.T[0][start_index:],outputs1.T[1][start_index:],outputs1.T[3][start_index:], zorder=1)
            ax3 = fig.add_subplot(223, projection='3d')
            ax3.set_xlabel("Neuron 1")
            ax3.set_ylabel("Neuron 3")
            ax3.set_zlabel("Neuron 4")
            nn = ctrnn.CTRNN(nnsize)
            nn.setParameters(target_genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
            nn.initializeState(np.zeros(nnsize))
            outputs = np.zeros((len(time),nnsize))
            step = 0
            for t in time:
                nn.step(stepsize)
                outputs[step] = nn.Output
                step += 1
            ax3.plot3D(outputs.T[0][start_index:],outputs.T[2][start_index:],outputs.T[3][start_index:],'k', label="Target", zorder=2)
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
                ax3.plot3D(outputs1.T[0][start_index:],outputs1.T[2][start_index:],outputs1.T[3][start_index:], zorder=1)
            ax4 = fig.add_subplot(224, projection='3d')
            ax4.set_xlabel("Neuron 2")
            ax4.set_ylabel("Neuron 3")
            ax4.set_zlabel("Neuron 4")
            nn = ctrnn.CTRNN(nnsize)
            nn.setParameters(target_genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
            nn.initializeState(np.zeros(nnsize))
            outputs = np.zeros((len(time),nnsize))
            step = 0
            for t in time:
                nn.step(stepsize)
                outputs[step] = nn.Output
                step += 1
            ax4.plot3D(outputs.T[1][start_index:],outputs.T[2][start_index:],outputs.T[3][start_index:],'k', label="Target", zorder=2)
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
                ax4.plot3D(outputs1.T[1][start_index:],outputs1.T[2][start_index:],outputs1.T[3][start_index:], zorder=1)
            plt.savefig("N{2}/{0}_cluster_{1}_phase.png".format(targ, color, nnsize))
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
    plt.savefig("N{1}/pca_{0}.png".format(targ, nnsize))
    df_PCA.to_csv("N{1}/parameter_pca_{0}.csv".format(targ, nnsize))
    
def pca_compare(params0, target0, params1, target1, num0, num1, size):
    param2 = np.zeros((params0.shape[0]+params1.shape[0]+2, params1.shape[1]))
    param2[:params0.shape[0]] = params0
    param2[-1] = target0
    param2[params0.shape[0]+1:-1] = params1
    param2[-1] = target1
    param3 = param2[:,:-size]
    indices = list(np.arange(params0.shape[0])) + ['*1'] + list(params0.shape[0]+np.arange(params1.shape[0])) + ['*2']
    pca = PCA(n_components=None, svd_solver='full')
    res = pca.fit_transform(param3)
    df_PCA = pd.DataFrame(res[:, 0:9], columns=['1c', '2c', '3c', '4c', '5c', '6c', '7c', '8c', '9c'], index=indices)
    s_Var = pd.Series(pca.explained_variance_ratio_, index=range(1, (res.shape[1] + 1)), name='explained_variance_ratio')
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(figsize=(12, 10), nrows=3, ncols=3)
    fig.suptitle("PCA on results for {0} neurons".format(size))
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
    plt.savefig("n{0}_double_pca_{1}_{2}.png".format(size, num0, num1))
    df_PCA.to_csv("N{0}/double_pca_{1}_{2}.csv".format(size, num0, num1))
    
def compare_phase(params, target_params):
    outs_list = []
    time = np.arange(0.0,duration,stepsize)
    for pars in params:
        nn = ctrnn.CTRNN(nnsize)
        nn.setParameters(pars,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
        nn.initializeState(np.zeros(nnsize))
        outputs = np.zeros((len(time),nnsize))
        step = 0
        for t in time:
            nn.step(stepsize)
            outputs[step] = nn.Output
            step += 1
        outs_list.append(outputs)
    for i in range(nnsize-2):
        if i%2 == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel("Neuron {0}".format(i))
            ax.set_ylabel("Neuron {0}".format(i+1))
            ax.set_zlabel("Neuron {0}".format(i+2))
            ax.plot3D(target_output.T[i][start_index:],target_output.T[i+1][start_index:],target_output.T[i+2][start_index:], 'k', zorder=2)#calculate target_output earlier
            for outs in outs_list:
                ax.plot3D(outs.T[i][start_index:],outs.T[i+1][start_index:],outs.T[i+2][start_index:], zorder=1)
            plt.savefig("N{0}/{2}/phase_{1}.png".format(nnsize, i, targ))
    
    pca = PCA(n_components=None, svd_solver='auto')
    pca.fit(target_output) #calculate target_output earlier
    res_target = pca.transform(target_output)
    res_list = []
    for outs in outs_list:
        res_est = pca.transform(outs)
        res_list.append(res_est)
#    df_PCA = pd.DataFrame(res_target[:, 0:9], columns=['1c', '2c', '3c', '4c', '5c', '6c', '7c', '8c', '9c'])
    df_PCA = pd.DataFrame(res_target[:, 0:4], columns=['1c', '2c', '3c', '4c'])
    s_Var = pd.Series(pca.explained_variance_ratio_, index=range(1, (res_target.shape[1] + 1)), name='explained_variance_ratio')
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(figsize=(12, 10), nrows=3, ncols=3)
    fig.suptitle("PCA on output for {0} neurons".format(nnsize))
    s_cumsum = s_Var.cumsum()
    n_eigen_95 = s_cumsum[(s_cumsum < 0.95)].shape[0]
    n = 4 #4
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
        ax.plot(xs[start_index:], ys[start_index:], 'k', zorder=2)
        ax.plot(0,0, color='#2ca02c', marker='x', ms=16)
        for res in res_list:
            ax.plot(res[start_index:,dim-1], res[start_index:,dim], zorder=1)
        ax.axhline(y=0, c='black', lw=0.75, ls='-.', zorder=2)
        ax.axvline(x=0, c='black', lw=0.75, ls='-.', zorder=2)
        ax.set_title('Components {dim1} and {dim2}'.format(dim1=dim, dim2=(dim + 1)) )
        ax.set_xlabel('Component {dim1:d}'.format(dim1=dim))
        ax.set_ylabel('Component {dim2:d}'.format(dim2=dim + 1))
        ax.grid()
        ax.axis('equal')
        ax.locator_params(axis='both', tight=True, nbins=6)
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.93, wspace=0.26, hspace=0.35)
    plt.savefig("N{1}/results_outputs_pca_{0}.png".format(targ, nnsize))
    df_PCA.to_csv("N{1}/results_outputs_pca_{0}.csv".format(targ, nnsize))
    
if __name__=='__main__':
    gens, gs, af, bf, bi = load_results(targ, nnsize, reps=50)
    
#    clust = dbscan(bi, bf[:,-1], target_genotype, 1, 3)    

    thr = np.mean(bf[:,-1]) + 0.35*np.std(bf[:,-1])
    bi1, bf1, af1 = filter_fit(bi, bf, af, thr)
    
#    x_embedded = TSNE(n_components=2, perplexity=5, learning_rate=200).fit_transform(bi)
#    plt.figure()
#    plt.scatter(x_embedded[:,0], x_embedded[:,1], color='g')
#    plt.title("t-SNE visualization - {0} neurons".format(nnsize))
#    plt.savefig("N{0}/tsne_{1}.png".format(nnsize, targ))
    
#    princ_comp(bi, target_genotype)
    compare_phase(bi1, target_genotype)
    
#    d1, c1, cl1 = dendro(bi1, target_genotype, 'weighted', 0.6)
#    plt.figure()
#    plt.title("{0} neurons".format(nnsize))
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
#    plt.savefig("N{1}/fit_vs_dist_{0}.png".format(targ, nnsize))
            
#    plt.figure()
#    plt.title("Average fitness")
#    plt.xlabel("Generations")
#    for ft in af:
#        plt.plot(np.arange(1, len(ft)+1), ft)
#    plt.savefig("N{0}/fitness_avg_{1}.png".format(nnsize, targ))
#    plt.figure()
#    plt.title("Best fitness")
#    plt.xlabel("Generations")
#    for ft in bf:
#        plt.plot(np.arange(1, len(ft)+1), ft)
#    plt.axhline(y=thr, color='r', linestyle='dashed')
#    plt.savefig("N{0}/fitness_best_{1}.png".format(nnsize, targ))
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
#    plt.savefig("N{1}/params_{0}".format(targ, nnsize))
    
#    estimated_pars = np.mean(bi1, axis=0)
#    plt.figure()
#    plt.scatter(np.arange(bi1.shape[1]-nnsize), 15*target_genotype[:-nnsize], color='r', label='Target')
#    plt.scatter(np.arange(bi1.shape[1]-nnsize), 15*estimated_pars[:-nnsize], color='g', label='Estimated')
#    plt.legend()
#    plt.ylabel("Parameter Value", fontsize=14)
#    plt.xlabel("Parameter Index", fontsize=14)
#    plt.title("Estimated parameter values (average)", fontsize=18)
#    plt.savefig("N{0}/avgestimate_{1}.png".format(nnsize, targ))
    
#    fig = plt.figure()
#    nn_1 = ctrnn.CTRNN(nnsize)
#    nn_1.setParameters(target_genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
#    nn_1.initializeState(np.zeros(nnsize))
#    time = np.arange(0.0,duration,stepsize)
#    outputs_targ = np.zeros((len(time),nnsize))
#    step = 0
#    for t in time:
#        nn_1.step(stepsize)
#        outputs_targ[step] = nn_1.Output
#        step += 1
#    nn_2 = ctrnn.CTRNN(nnsize)
#    nn_2.setParameters(estimated_pars,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
#    nn_2.initializeState(np.zeros(nnsize))
#    time = np.arange(0.0,duration,stepsize)
#    outputs_est = np.zeros((len(time),nnsize))
#    step = 0
#    for t in time:
#        nn_2.step(stepsize)
#        outputs_est[step] = nn_2.Output
#        step += 1
#    if nnsize == 3:
#        ax = fig.add_subplot(111, projection='3d')
#        plt.title("Phase portrait")
#        ax.set_xlabel("Neuron 1")
#        ax.set_ylabel("Neuron 2")
#        ax.set_zlabel("Neuron 3")
#        ax.plot3D(outputs_targ.T[0][start_index:],outputs_targ.T[1][start_index:],outputs_targ.T[2][start_index:],'r', label="Target",zorder=1)
#        ax.plot3D(outputs_est.T[0][start_index:],outputs_est.T[1][start_index:],outputs_est.T[2][start_index:],'g', label="Estimated",zorder=2)
#        ax.legend()
#        plt.savefig("N{0}/avgestimate_phase_{1}.png".format(nnsize, targ))
#    elif nnsize == 4:
#        plt.suptitle("Phase portrait")
#        ax1 = fig.add_subplot(131, projection='3d')
#        ax1.set_xlabel("Neuron 1")
#        ax1.set_ylabel("Neuron 2")
#        ax1.set_zlabel("Neuron 3")
#        ax1.plot3D(outputs_targ.T[0][start_index:],outputs_targ.T[1][start_index:],outputs_targ.T[2][start_index:],'r', label="Target",zorder=1)
#        ax1.plot3D(outputs_est.T[0][start_index:],outputs_est.T[1][start_index:],outputs_est.T[2][start_index:],'g', label="Estimated",zorder=2)
#        ax2 = fig.add_subplot(132, projection='3d')
#        ax2.set_xlabel("Neuron 1")
#        ax2.set_ylabel("Neuron 2")
#        ax2.set_zlabel("Neuron 4")
#        ax2.plot3D(outputs_targ.T[0][start_index:],outputs_targ.T[1][start_index:],outputs_targ.T[3][start_index:],'r', label="Target",zorder=1)
#        ax2.plot3D(outputs_est.T[0][start_index:],outputs_est.T[1][start_index:],outputs_est.T[3][start_index:],'g', label="Estimated",zorder=2)
#        ax3 = fig.add_subplot(133, projection='3d')
#        ax3.set_xlabel("Neuron 2")
#        ax3.set_ylabel("Neuron 3")
#        ax3.set_zlabel("Neuron 4")
#        ax3.plot3D(outputs_targ.T[1][start_index:],outputs_targ.T[2][start_index:],outputs_targ.T[3][start_index:],'r', label="Target",zorder=1)
#        ax3.plot3D(outputs_est.T[1][start_index:],outputs_est.T[2][start_index:],outputs_est.T[3][start_index:],'g', label="Estimated",zorder=2)
#        plt.tight_layout()
#        plt.savefig("n{0}_avgestimate_phase_{1}.png".format(nnsize, targ))
    