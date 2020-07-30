
import ctrnn
import numpy as np
import matplotlib.pyplot as plt
import sys


dir = 'C:\\Users\\Sikander\\Documents\\Indiana University\\Classes\\I698 - Research in Informatics\\inferweights\\'#str(sys.argv[0])
reps = 1#int(sys.argv[1])
stepsize = 0.01#float(sys.argv[2])
fitness_threshold = 0.7#float(sys.argv[3])

# NN Params
nnsize = 2
duration = 50
start_index = int(45/stepsize)
WeightRange = 15
BiasRange = 15
TimeConstMin = 1.0 #stepsize*10
TimeConstMax = 1.0 #2.0

gens = len(np.load(dir+"avg_fitness.npy"))
gs=len(np.load(dir+"best_individual.npy"))
af = np.zeros((reps,gens))
bf = np.zeros((reps,gens))
bi = np.zeros((reps,gs))
for i in range(reps):
    af[i] = np.load(dir+"avg_fitness.npy") #used be "avg_fitness_"+str(i)+".npy"
    bf[i] = np.load(dir+"best_fitness.npy")
    bi[i] = np.load(dir+"best_individual.npy")

plt.plot(af.T,'b')
plt.plot(bf.T,'g')
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("Evolution")
plt.show()

plt.figure()
 # Get best evolved networks and show its activity
for i in range(reps):
    if bf[i][-1]>fitness_threshold:
        time = np.arange(0.0,duration,stepsize)
        nn = ctrnn.CTRNN(nnsize)
        nn.setParameters(bi[i],WeightRange,BiasRange,TimeConstMin,TimeConstMax)
        nn.initializeState(np.zeros(nnsize))
        outputs = np.zeros((len(time),nnsize))
        step = 0
        for t in time:
            nn.step(stepsize)
            outputs[step] = nn.Output
            step += 1
        plt.plot(outputs.T[0][start_index:],outputs.T[1][start_index:])
        
# Also show the activity of the target network
target_genotype = np.load(dir+"target_genotype.npy")
time = np.arange(0.0,duration,stepsize)
nn = ctrnn.CTRNN(nnsize)
nn.setParameters(bi[i],WeightRange,BiasRange,TimeConstMin,TimeConstMax)
nn.initializeState(np.zeros(nnsize))
outputs = np.zeros((len(time),nnsize))
step = 0
for t in time:
    nn.step(stepsize)
    outputs[step] = nn.Output
    step += 1
plt.plot(outputs.T[0][start_index:],outputs.T[1][start_index:],'k')

plt.xlabel("Neuron 1 output")
plt.ylabel("Neuron 2 output")
plt.title("Phase Portrait")
plt.show()

def eucdist(x,y):
    return np.sum((x-y)**2)**(1/2)

      
    
    
# Calculate and show the difference between the target parameters and those of the evolved circuits
rel = []
for i in range(reps):
    if bf[i][-1]>fitness_threshold:
        finalfit=bf[i][-1]
        paramdist=eucdist(bi[i][:6],target_genotype[:6])
        print(i,finalfit,paramdist)
        rel.append([finalfit,paramdist])
rel = np.array(rel)
plt.figure()
plt.plot(rel.T[0],rel.T[1],'o')
plt.xlabel("Final fitness")
plt.ylabel("Param dist to target")
plt.show()
