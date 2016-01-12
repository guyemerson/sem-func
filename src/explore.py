import numpy
from numpy import zeros
numpy.set_printoptions(precision=2, suppress=True)
import pickle
from model import SemFuncModel

with open('../data/out.pkl', 'rb') as f:
    m = pickle.load(f)

cos = zeros((m.V, m.V))

for i in range(m.V):
    for j in range(m.V):
        cos[i,j] = m.cosine_of_parameters(i,j)

print(cos)

cos = zeros((m.V, m.V))

for i in range(m.V):
    for j in range(m.V):
        cos[i,j] = m.cosine_samples(i,j, samples=50)

print(cos)