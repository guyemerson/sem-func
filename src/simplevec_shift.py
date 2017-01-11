import pickle, gzip, os

# Hyperparameters

dataset = 'multicore-5'
D = 400  # Number of dimensions for nouns and verbs (separately) 
a = 0.8  # Power that frequencies are raised to under the null hypothesis
seed = 91

k_range = [0.5, 0.8, 1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2, 2.3, 2.6, 3]

# Load original file

full_template = '/anfs/bigdisc/gete2/wikiwoods/simplevec/{}-{}-{}-{}-{}.pkl.gz'
template = full_template.format(dataset, D, '{}', str(a).replace('.',''), seed)

with gzip.open(template.format(0), 'rb') as f:
    vec = pickle.load(f)

# Shift and save


for k in k_range:
    filename = template.format(str(k).replace('.',''))
    if os.path.exists(filename): continue
    print(k)
    new = (vec - k).clip(0)
    with gzip.open(filename, 'wb') as f:
        pickle.dump(new, f)
