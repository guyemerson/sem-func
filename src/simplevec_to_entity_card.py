import os
from itertools import product
from multiprocessing import Pool
from random import shuffle

from simplevec_to_entity import get_entities
from __config__.filepath import AUX_DIR

# Grid search over hyperparameters

scales = [0.8, 1, 1.2]
Zs = [0.0001, 0.001, 0.01]
alphas = [0, 0.6, 0.7, 0.75, 0.8, 0.9, 1]

grid = product(scales, Zs, alphas)

# Vector models

simplevec = os.listdir(os.path.join(AUX_DIR, 'simplevec_card'))
simplevec_filtered = []
for name in simplevec:
    parts = name.split('-')
    if len(parts) != 7:
        continue
    prefix, thresh, dim, C, k, a, seed = parts
    #if prefix == 'multicore' and thresh == '5' and dim == '400' and k == '0' and a in ['075','08','09','1']:
    simplevec_filtered.append((name.split('.')[0], int(C)))

full_grid = list(product(grid, simplevec_filtered))
shuffle(full_grid)

def train(hyper, simple_and_C):
    scale, Z, alpha = hyper
    simple, C = simple_and_C
    print(scale, Z, alpha, simple, C)
    get_entities('frequency', scale, C, Z=Z, alpha=alpha, name=simple, mean_field_kwargs={"max_iter":500}, output_dir='meanfield_freq_card', input_dir='simplevec_card')

with Pool(32) as p:
    p.starmap(train, full_grid)
