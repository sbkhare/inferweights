# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:41:19 2020

@author: Sikander
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from collections import Counter
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import ctrnn
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import pandas as pd
import sys

stepsize=0.01
fitness_threshold = 0.7
# NN Params
nnsize = 2
duration = 100
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
        plt.savefig("{0}_cluster_{1}_genotype.png".format(targ, color))
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
            plt.savefig("{0}_cluster_{1}_phase.png".format(targ, color))
    return d, clusters, clust_list

if __name__=='__main__':
#    clust = dbscan(bi, bf[:,-1], target_genotype, 1, 3)    

    
#   
    thr = np.mean(bf[:,-1]) #+ np.std(bf[:,-1])
    bi1, bf1, af1 = filter_fit(bi, bf, af, thr)
    
    d1, c1, cl1 = dendro(bi1, target_genotype, 'complete', 0.7)
#    d2 = dendro(bi, target_genotype, 'ward')
    plt.figure()
    plt.title("2 neurons")
    plt.xlabel("Fitness")
    plt.ylabel("Parameter distance")
    x = []
    y = []
    for clust in c1:
        distlist = []
        fitlist = []
        pt_label = []
        i = 0
        for gene in c1[clust]:
            bf_idx = cl1[clust][i]
            i += 1
            if bf_idx == '*':
                continue
            else:
                fitlist.append(bf1[bf_idx,-1])
                pt_label.append(bf_idx)
                x.append(bf1[bf_idx,-1])
                dist = eucdist(target_genotype, gene)
                distlist.append(dist)
                y.append(dist)
        plt.scatter(fitlist, distlist, c=clust)
        for j, txt in enumerate(pt_label):
            plt.annotate(txt, (fitlist[j], distlist[j]))
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), c='k', linestyle='dashed') #ax+b; coeff = np.polyfit(x,y,1), a=coeff[0], b=coeff[1]
    plt.savefig("fit_vs_dist_{0}.png".format(targ))
#    
#    plt.figure()
#    plt.title("Average fitness")
#    plt.xlabel("Generations")
#    for ft in af:
#        plt.plot(np.arange(1, len(ft)+1), ft)
#    plt.savefig("fitness_avg.png")
    
    
#    plt.figure()
##    plt.subplot(121)
##    plt.title("Fitness of Best Individuals", fontsize=14)
#    plt.ylabel("Fitness", fontsize=18)
#    plt.xlabel("Generation", fontsize=18)
#    plt.xticks(fontsize=14)
#    plt.yticks(fontsize=14)
#    for ft in bf:
#        plt.plot(np.arange(1, len(ft)+1), ft, 'b', alpha=0.5)
#    plt.axhline(y=thr, color='r', linestyle='dashed', linewidth=2)
#    plt.tight_layout()
##    plt.savefig("fitness_best.png")
#    plt.figure()
##    plt.subplot(122)
##    plt.title("Limit Cycle in Phase Space", fontsize=14)
#    plt.xlabel("Neuron 1 Output", fontsize=18)
#    plt.ylabel("Neuron 2 Output", fontsize=18)
#    plt.xticks(fontsize=14)
#    plt.yticks(fontsize=14)
#    plt.tight_layout()
#    nn = ctrnn.CTRNN(nnsize)
#    nn.setParameters(target_genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
#    nn.initializeState(np.zeros(nnsize))
#    time = np.arange(0.0,duration,stepsize)
#    outputs = np.zeros((len(time),nnsize))
#    step = 0
#    for t in time:
#        nn.step(stepsize)
#        outputs[step] = nn.Output
#        step += 1
#    plt.plot(outputs.T[0][start_index:],outputs.T[1][start_index:], 'k', label="Target", zorder=2)
#    for pars in bi1:
#        nn1 = ctrnn.CTRNN(nnsize)
#        nn1.setParameters(pars,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
#        nn1.initializeState(np.zeros(nnsize))
#        outputs1 = np.zeros((len(time),nnsize))
#        step1 = 0
#        for t in time:
#            nn1.step(stepsize)
#            outputs1[step1] = nn1.Output
#            step1 += 1
#        plt.plot(outputs1.T[0][start_index:],outputs1.T[1][start_index:], zorder=1, alpha=1)
##    plt.legend()
    clust_arr = np.zeros(bi1.shape[0], dtype=str)
    for clust in cl1:
        for ind in cl1[clust]:
            if ind == '*':
                continue
            else:
                clust_arr[ind] = clust
    colnum = bi1.shape[1]
    bi_cat = np.concatenate(bi1)
    data = pd.DataFrame([[15*target_genotype[0], '0', "Target"]], columns=["Value", "Gene", "Cluster"])
    for idx, val in enumerate(target_genotype):
        if idx == 0:
            continue
        else:
            if idx%colnum == 6 or idx%colnum == 7:
#                data = data.append({"Value": val, "Gene": str(idx%colnum), "Type": "Target"}, ignore_index=True)
                continue
            else:
                data = data.append({"Value": 15*val, "Gene": str(idx%colnum), "Cluster": "Target"}, ignore_index=True)
    for idx, val in enumerate(bi_cat):
        if idx%colnum == 6 or idx%colnum == 7:
#            data = data.append({"Value": 15*val, "Gene": str(idx%colnum), "Type": "Inferred"}, ignore_index=True)
            continue
        else:
            data = data.append({"Value": 15*val, "Gene": str(idx%colnum), "Cluster": clust_arr[idx//colnum]}, ignore_index=True)
    plt.figure()
#    plt.title("Parameter Values", fontsize=14)
    g = sns.swarmplot(x="Gene", y="Value", hue="Cluster", data=data) #, palette="Set2"
    plt.ylabel("Parameter Value", fontsize=18)
    plt.xlabel("Parameter Index", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("params_{0}".format(targ))
#   