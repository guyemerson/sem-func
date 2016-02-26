import numpy
from numpy import zeros, tensordot
numpy.set_printoptions(precision=3, suppress=True, threshold=numpy.nan)
from math import exp
import pickle
from model import SemFuncModel
from pydmrs.core import DictPointDmrs as Dmrs, PointerNode as Node, Link

with open('../data/out.pkl', 'rb') as f:
    m = pickle.load(f)

print("=== Link weights ===\n")
print(m.link_wei)
x = m.link_wei.copy()
x[x<0.09] = 0
print("\n=== Thresholded link weights ===\n")
print(x)
print("\n=== Pred weights ===\n")
print(m.pred_wei)


noun = ['dog','duck', 'pigeon']
verb = ['run','swim','fly','eat','breathe','bark','growl','quack','coo']
N, V = len(noun), len(verb)

energy = zeros((V,N))

for i in range(V):
    for j in range(N):
        nodes = [Node(0,N+i),
                 Node(1,j)]
        links = [Link(0,1,0,None)]
        dmrs = Dmrs(nodes, links)
        energy[i,j] = m.sample_energy(dmrs)
print("\n=== Co-occurrence predictions ===\n")
print(energy)

sent = [[1,0,1],
        [1,1,0],
        [0,1,1],
        [1,1,1],
        [1,1,1],
        [1,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1]]

right_energy = [energy[i,j] for i in range(V) for j in range(N) if sent[i][j]]
wrong_energy = [energy[i,j] for i in range(V) for j in range(N) if not sent[i][j]]
print("right:", sum(right_energy)/len(right_energy))
print("wrong:", sum(wrong_energy)/len(wrong_energy))


cos = zeros((m.V, m.V))

for i in range(m.V):
    for j in range(m.V):
        cos[i,j] = m.cosine_of_parameters(i,j)
print("\n=== Cosine of parameters ===\n")
print(cos)

cos = zeros((m.V, m.V))

for i in range(m.V):
    for j in range(m.V):
        cos[i,j] = m.cosine_samples(i,j, samples=5)

print("\n=== Cosine of samples ===\n")
print(cos)


print("\n=== Sample! ===\n")

ents = zeros((2, m.D))
nodes = [Node(0, None), Node(1, None)]
links = [Link(0, 1, 0, None)]
dmrs = Dmrs(nodes, links)

for _ in range(10):
    m.resample(nodes, ents, pred=False)

probtotal = zeros((m.V, m.V))
n = 0

while True:
    m.resample(nodes, ents, pred=False)
    raw = zeros((m.V, m.V))
    for i in range(m.V):
        for j in range(m.V):
            
            # Sem-func model
            raw[i,j] = m.prob(ents[0], i) * m.freq[i] \
                      *m.prob(ents[1], j) * m.freq[j]
            """
            # Classifier model
            raw[i,j] = exp(tensordot(m.pred_wei[i, :],
                                      ents[0], (0,0))
                          +tensordot(m.pred_wei[j, :],
                                      ents[1], (0,0))) \
                      * freq[i] * freq[j]
            """
            
    probtotal += raw / numpy.sum(raw)
    n += 1
    print(probtotal / n)

