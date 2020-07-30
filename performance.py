# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:53:02 2020

@author: Sikander
"""
import numpy as np
import matplotlib.pyplot as plt

n2_trials = [12,13,14]
n3_trials = [13,14,15]
n4_trials = [1,2,4]

def agg_fitness(neur_num, trials, clr):
    reps = 50
    gens = len(np.load("N{0}/{1}/avg_fitness_{1}_0.npy".format(neur_num, trials[0])))
    gs = len(np.load("N{0}/{1}/best_individual_{1}_0.npy".format(neur_num, trials[0])))
#    af = np.zeros((reps,gens))
    bf = np.zeros((reps,gens))
#    bi = np.zeros((reps,gs))
    for j, t in enumerate(trials):
        if j == 0:
            for i in range(reps):
#                af[i] = np.load("N{0}/{1}/avg_fitness_{1}_{2}.npy".format(neur_num, t, i)) 
                bf[i] = np.load("N{0}/{1}/best_fitness_{1}_{2}.npy".format(neur_num, t, i))
#                bi[i] = np.load("N{0}/{1}/best_individual_{1}_{2}.npy".format(neur_num, t, i))
        else:
#            af1 = np.zeros((reps,gens))
            bf1 = np.zeros((reps,gens))
#            bi1 = np.zeros((reps,gs))
            for i in range(reps):
#                af1[i] = np.load("N{0}/{1}/avg_fitness_{1}_{2}.npy".format(neur_num, t, i)) 
                bf1[i] = np.load("N{0}/{1}/best_fitness_{1}_{2}.npy".format(neur_num, t, i))
#                bi1[i] = np.load("N{0}/{1}/best_individual_{1}_{2}.npy".format(neur_num, t, i))
#            af = np.vstack((af, af1))
            bf = np.vstack((bf, bf1))
#            bi = np.vstack((bi, bi1))
    avg_bf = np.mean(bf, axis=0)
    ci_bf = 2*np.std(bf, axis=0)
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(avg_bf)), avg_bf, color=clr)
    ax.fill_between(np.arange(len(avg_bf)), (avg_bf-ci_bf), (avg_bf+ci_bf), color=clr, alpha=0.1)
    ax.set_xlabel("Generation", fontsize=16)
    ax.set_ylabel("Fitness of best individual", fontsize=16)
#    ax.set_title("Best fitness performance for {0}-neuron system".format(neur_num), fontsize=14)
    ax.grid()
    return fig
    
def eucdist(x,y):
    return np.sum((x-y)**2)**(1/2)

def pardist_fits(neur_nums, trial_lsts):
    x = []
    y = []
    yerr = []
    y2 = []
    yerr2 = []
    for num, trials in zip(neur_nums, trial_lsts):
        x.append(num)
        dist_lst = []
        bf_lst = []
        for t in trials:
            targ = np.load("N{0}/{1}/target_genotype_{1}.npy".format(num, t))
            reps = 50
            for i in range(reps):
                g = np.load("N{0}/{1}/best_individual_{1}_{2}.npy".format(num, t, i))
                bf = np.load("N{0}/{1}/best_fitness_{1}_{2}.npy".format(num, t, i))
                dist_lst.append(eucdist(targ, g))
                bf_lst.append(bf[-1])
        y.append(np.mean(dist_lst))
        yerr.append(np.std(dist_lst))
        y2.append(np.mean(bf_lst))
        yerr2.append(np.std(bf_lst))
    plt.figure()
    plt.plot(x, y, marker="D") #plt.bar yerr=yerr, ecolor='black', capsize=10, alpha=0.5)
    plt.errorbar(x, y, yerr, capsize=10) #uplims=True, lolims=True
    plt.xlabel("Number of neurons", fontsize=16)
    plt.xticks([2,3,4], [2,3,4], fontsize=14)
    plt.yticks(fontsize=14)
#    plt.title("Distance from target parameters", fontsize=14)
    plt.ylabel("Parameter distance", fontsize=16)
    plt.tight_layout()
    plt.figure()
    plt.plot(x, y2, marker="D") #plt.bar yerr=yerr, ecolor='black', capsize=10, alpha=0.5)
    plt.errorbar(x, y2, yerr2, capsize=10) #uplims=True, lolims=True
    plt.xlabel("Number of neurons", fontsize=16)
    plt.xticks([2,3,4], [2,3,4], fontsize=14)
    plt.yticks([0.7 , 0.75, 0.8 , 0.85, 0.9 ], [0.7 , 0.75, 0.8 , 0.85, 0.9 ], fontsize=14)
    plt.yticks(fontsize=14)
#    plt.title("Best fitness")
    plt.ylabel("Fitness", fontsize=16)
    plt.tight_layout()
    print(y, yerr, y2, yerr2)
            
        
    
if __name__=='__main__':
#    fig1 = agg_fitness(2, n2_trials, 'b')
#    fig2 = agg_fitness(3, n3_trials, 'g')
#    fig3 = agg_fitness(4, n4_trials, 'r')
#    plt.figure()
    pardist_fits([2,3,4], [n2_trials, n3_trials, n4_trials])