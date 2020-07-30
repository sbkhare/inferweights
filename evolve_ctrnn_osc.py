import mga
import ctrnn
import numpy 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

num = sys.argv[1]
# NN Params
nnsize = int(sys.argv[2])
duration = 50
stepsize = 0.01
WeightRange = 15
BiasRange = 15
TimeConstMin = 1.0 #stepsize*10
TimeConstMax = 1.0 #2.0
start_index = int(42/stepsize)

# Fitness function
def fitnessFunction(genotype):
    time = numpy.arange(0.0,duration,stepsize)
    nn = ctrnn.CTRNN(nnsize)
    nn.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
    nn.initializeState(numpy.zeros(nnsize))
    fit = 0.0
    for t in time:
        pastOutputs = nn.Output
        nn.step(stepsize)
        currentOutputs = nn.Output
        fit += numpy.sum(abs(currentOutputs - pastOutputs))
    return fit/(nnsize*duration)

# EA Params
popsize = 500
demesize = 10
genesize = nnsize*nnsize + 2*nnsize
recombProb = 0.5
mutatProb = 1/genesize
generations = 150

# Evolve and visualize fitness over generations
ga = mga.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demesize, generations, 1)
ga.run()
ga.showFitness()

# Get best evolved network and show its activity
af,bf,bi = ga.fitStats()
time = numpy.arange(0.0,duration,stepsize)
nn = ctrnn.CTRNN(nnsize)
nn.setParameters(bi,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
nn.initializeState(numpy.zeros(nnsize))
outputs = numpy.zeros((len(time),nnsize))
step = 0
for t in time:
    nn.step(stepsize)
    outputs[step] = nn.Output
    step += 1
plt.figure()
for i in range(nnsize):
    plt.plot(time,outputs)
plt.xlabel("Time")
plt.ylabel("Output")
plt.title("Neural activity")
plt.show()
plt.savefig("n{0}/{1}/neuralactivity.png".format(nnsize,num))


for i in range(nnsize-2):
    if nnsize%2 == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("Neuron {0}".format(i))
        ax.set_ylabel("Neuron {0}".format(i+1))
        ax.set_zlabel("Neuron {0}".format(i+2))
        ax.plot3D(outputs.T[i][start_index:],outputs.T[i+1][start_index:],outputs.T[i+2][start_index:])
        plt.savefig("n{0}/{2}/phase_{1}.png".format(nnsize, i, num))
        

#file_outputs = open("target_output_{0}.npy".format(num), 'rb')
#numpy.save(file_outputs, outputs)
#file_outputs.flush()
#file_genotype = open("target_genotype_{0}.npy".format(num), 'rb')
#numpy.save(file_genotype, bi)
#file_genotype.flush()

numpy.save("target_output_{0}.npy".format(num),outputs)
numpy.save("target_genotype_{0}.npy".format(num),bi)
