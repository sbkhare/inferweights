# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:37:50 2020

@author: Sikander
"""
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode 

target_output = np.load("N2/12/target_output_12.npy")
target_output = target_output[:1000,:]

def targetPeriod(outs):
    #Return target period and start of cycle
    ind_dct = {}
    count1 = 0
    for o in target_output:
        count1 += 1
        count2 = 0
        ind_lst = []
        for t in target_output:
            count2 += 1 
            if np.linalg.norm(t - o) < 0.01:
                ind_lst.append(count2-1)
        if len(ind_lst) > 0:
            ind_dct[count1-1] = ind_lst
    #return ind_dct
    period_lst = []
    period_dct = {}
    for ind in ind_dct:
        periods = np.diff(ind_dct[ind])
        periods = list(filter(lambda a: a != 1, periods))
        period_lst = period_lst + periods
        period_dct[ind] = periods
#    plt.figure()
#    plt.hist(period_lst, 50)
#    plt.show()
    avg = np.mean(period_lst)
    max_T = max(period_lst)
    mode_T = mode(period_lst)
    print("Average period: {0}".format(avg))
    print("Max period: {0}".format(max_T))
    print("Most common period: {0}".format(mode_T))
    start_ind = 0 #[]
#    num = 0
    for ind in period_dct:
        if mode_T in period_dct[ind]:
            print("Start index: {0}".format(ind))
            start_ind = ind #Maybe make it the fourth/fifth appearance of the period?
            break
            #start_ind.append(ind) 
#        if num == cyc_num:
#            break
#        num += 1
    return ind_dct, period_dct, mode_T, start_ind

if __name__=='__main__':
    ind_dct, prd_dct, prd, start = targetPeriod(target_output)
    start_index = start
    for period in prd_dct[start]:
        start_index += period
    start_index -= prd_dct[start][-1]
    
    plt.figure()
    plt.plot(target_output[:,0], target_output[:,1])
    plt.plot(target_output[start,0], target_output[start,1], marker='o', markersize=7, color='red')
    plt.plot(target_output[start_index,0], target_output[start_index,1], marker='o', markersize=7, color='blue')
    plt.xlabel("Neuron 1 output")
    plt.ylabel("Neuron 2 output")
    plt.title("Phase Portrait")
    plt.show()
    
#    plt.figure()
#    plt.plot(target_output[:,2], target_output[:,1])
#    plt.plot(target_output[start,2], target_output[start,1], marker='o', markersize=7, color='red')
#    plt.plot(target_output[start_index,2], target_output[start_index,1], marker='o', markersize=7, color='blue')
#    plt.xlabel("Neuron 3 output")
#    plt.ylabel("Neuron 2 output")
#    plt.title("Phase Portrait")
#    plt.show()
#    
#    plt.figure()
#    plt.plot(target_output[:,0], target_output[:,2])
#    plt.plot(target_output[start,0], target_output[start,2], marker='o', markersize=7, color='red')
#    plt.plot(target_output[start_index,0], target_output[start_index,2], marker='o', markersize=7, color='blue')
#    plt.xlabel("Neuron 1 output")
#    plt.ylabel("Neuron 3 output")
#    plt.title("Phase Portrait")
#    plt.show()
#    
#    plt.figure()
#    plt.subplot(131)
#    plt.plot(target_output[start_index:,0], target_output[start_index:,1])
#    plt.xlabel("Neuron 1 output")
#    plt.ylabel("Neuron 2 output")
#    plt.subplot(132)
#    plt.plot(target_output[start_index:,2], target_output[start_index:,1])
#    plt.xlabel("Neuron 3 output")
#    plt.ylabel("Neuron 2 output")
#    plt.subplot(133)
#    plt.plot(target_output[start_index:,0], target_output[start_index:,2])
#    plt.xlabel("Neuron 1 output")
#    plt.ylabel("Neuron 3 output")
#    plt.suptitle("Limit Cycles")
#    plt.show()