import numpy
from numpy import zeros
numpy.set_printoptions(precision=2, suppress=True, threshold=numpy.nan)
import pickle
from model import SemFuncModel

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