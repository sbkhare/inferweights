import mga
import ctrnn
import numpy 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import sys

targ = sys.argv[1]
num = sys.argv[2]
# NN Params
nnsize = 4
duration = 50
stepsize = 0.01
start_ind = int(46/stepsize)
WeightRange = 15
BiasRange = 15
TimeConstMin = 1.0
TimeConstMax = 1.0 

target_output = numpy.load("target_output_{0}.npy".format(targ))
print(target_output.shape)
#ind_dct, prd_dct, prd, start = targetPeriod(target_output)
#start_index = start #For target
#start_ind = start #For microbial output
#for period in prd_dct[start]:
#    if start_ind + prd > int(duration/stepsize):
#        start_index += period
#    else:
#        start_index += period
#        start_ind += period
#start_index -= prd_dct[start][-1]
#start_ind -= 2*prd
#start_ind = int(7/stepsize)

pca = PCA(n_components=None, svd_solver='auto')

pca.fit(target_output)
res_targ = pca.transform(target_output)

# Fitness function: Difference between phase portraits
def fitnessFunction(genotype):
    time = numpy.arange(0.0,duration,stepsize)
    net = ctrnn.CTRNN(nnsize)
    net.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
    net.initializeState(numpy.zeros(nnsize))
    outputs = numpy.zeros((len(time),nnsize))
    for t in time:
        net.step(stepsize)
        outputs[int(t/stepsize)] = net.Output
    # Find all the pairwise distances between each of the points in the two systems across time (ignoring the transient)
    dist = cdist(outputs[start_ind:], target_output[start_ind:], metric='euclidean')
    # Find the one point where the distance between the two systems was the smallest: t1 and t2
    t1,t2 = numpy.unravel_index(numpy.argmin(dist, axis=None), dist.shape)
    # Use that point to align the rest of the points and take the cumulative of that distance
    cum_dist = 0.0
    n=len(dist)
    for i in range(n):
        cum_dist += dist[t1][t2]
        t1 = (t1+1)%n
        t2 = (t2+1)%n
    avg_dist = cum_dist/(n*numpy.sqrt(nnsize))
    return 1-avg_dist

def large_n_fitness(genotype):
    time = numpy.arange(0.0,duration,stepsize)
    net = ctrnn.CTRNN(nnsize)
    net.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
    net.initializeState(numpy.zeros(nnsize))
    outputs = numpy.zeros((len(time),nnsize))
    for t in time:
        net.step(stepsize)
        outputs[int(t/stepsize)] = net.Output
     #CONVERT OUTPUTS INTO PCA OUTPUTS BY APPLYING same transformation
    
    res_out = pca.transform(outputs)
    dim = 2
     # Find all the pairwise distances between each of the points in PCA space in the two systems across time (ignoring the transient)
    dist = cdist(res_out[start_ind:,:dim], res_targ[start_ind:,:dim], metric='euclidean')
    # Find the one point where the distance between the two systems was the smallest: t1 and t2
    t1,t2 = numpy.unravel_index(numpy.argmin(dist, axis=None), dist.shape)
    # Use that point to align the rest of the points and take the cumulative of that distance
    cum_dist = 0.0
    n=len(dist)
    for i in range(n):
        cum_dist += dist[t1][t2]
        t1 = (t1+1)%n
        t2 = (t2+1)%n
    avg_dist = cum_dist/(n*numpy.sqrt(dim))
    return 1-avg_dist
    

# EA Params
popsize = 500
demesize = 5
genesize = nnsize*nnsize + 2*nnsize
recombProb = 0.5
mutatProb = 1/genesize
generations = 1500

# Evolve and visualize fitness over generations
ga = mga.Microbial(large_n_fitness, popsize, genesize, recombProb, mutatProb, demesize, generations, 1)
ga.run()
#ga.showFitness()
af,bf,bi = ga.fitStats()

#id = int(sys.argv[1])
numpy.save("avg_fitness_{0}_{1}.npy".format(targ, num), ga.avgHistory) #"avg_fitness_"+str(id)+".npy"
numpy.save("best_fitness_{0}_{1}.npy".format(targ, num), ga.bestHistory) #"best_fitness_"+str(id)+".npy"
numpy.save("best_individual_{0}_{1}.npy".format(targ, num), bi) #"best_individual_"+str(id)+".npy"


#avg_fit = open("avg_fitness_{0}.npy".format(num), 'rb')
#numpy.save(avg_fit, ga.avgHistory)
#avg_fit.flush()
#best_fit = open("best_fitness_{0}.npy".format(num), 'rb')
#numpy.save(avg_fit, ga.bestHistory)
#best_fit.flush()
#best_indiv = open("best_individual_{0}.npy".format(num), 'rb')
#numpy.save(best_indiv, bi)
#best_indiv.flush()


# # Get best evolved network and show its activity
# time = np.arange(0.0,duration,stepsize)
# nn = ctrnn.CTRNN(nnsize)
# nn.setParameters(bi,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
# nn.initializeState(np.zeros(nnsize))
# outputs = np.zeros((len(time),nnsize))
# step = 0
# for t in time:
#     nn.step(stepsize)
#     outputs[step] = nn.Output
#     step += 1
# plt.figure()
# for i in range(nnsize):
#     plt.plot(time,outputs)
# plt.xlabel("Time")
# plt.ylabel("Output")
# plt.title("Neural activity")
# plt.show()
