import numpy
from numpy import zeros
numpy.set_printoptions(precision=2, suppress=True, threshold=numpy.nan)
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


noun = ['dog','cat','mouse']
verb = ['chase']

energy = zeros((3,3))

for i in range(3):
    for j in range(3):
        nodes = [Node(0,3),
                 Node(1,i),
                 Node(2,j)]
        links = [Link(0,1,0,None),
                 Link(0,2,1,None)]
        dmrs = Dmrs(nodes, links)
        energy[i,j] = m.sample_energy(dmrs)
print("\n=== Co-occurrence predictions ===\n")
print(energy)


cos = zeros((m.V, m.V))

for i in range(m.V):
    for j in range(m.V):
        cos[i,j] = m.cosine_of_parameters(i,j)
print("\n=== Cosine of parameters ===\n")
print(cos)

cos = zeros((m.V, m.V))

for i in range(m.V):
    for j in range(m.V):
        cos[i,j] = m.cosine_samples(i,j, samples=50)

print("\n=== Cosine of samples ===\n")
print(cos)


print("\n=== Sample! ===\n")

nodes = [Node(0, None), Node(1, None), Node(2, None)]
links = [Link(0, 1, 0, None), Link(0, 2, 1, None)]
dmrs = Dmrs(nodes, links)



badtotal = 0
probtotal = zeros((m.V-1, m.V-1))
n = 0

while True:
    ents = numpy.random.binomial(1, m.C/m.D, (3, m.D))
    """
    ents = numpy.zeros((3,m.D))
    ents[0][8] = 1; ents[0][15] = 1; ents[0][16] = 1
    ents[1][18] = 1; ents[1][5] = 1; ents[1][7] = 1
    ents[2][11] = 1; ents[2][12] = 1; ents[2][2] = 1
    """
    for _ in range(100):
        m.resample(nodes, ents, pred=False)
    for _ in range(100):
        m.resample(nodes, ents, pred=False)
        verbprob = m.prob(ents[0], 3)
        badtotal += 1-verbprob
        raw = zeros((m.V-1, m.V-1))
        for i in range(m.V-1):
            for j in range(m.V-1):
                
                # Sem-func model
                raw[i,j] = m.prob(ents[1], i) * m.freq[i] \
                          *m.prob(ents[2], j) * m.freq[j]
                """
                # Classifier model
                raw[i,j] = exp(tensordot(m.pred_wei[i, :],
                                          ents[0], (0,0))
                              +tensordot(m.pred_wei[j, :],
                                          ents[1], (0,0))) \
                          * freq[i] * freq[j]
                """
                
        probtotal += verbprob * raw / numpy.sum(raw)
        n += 1
        print(probtotal / n, badtotal / n)
