import os, pickle, gzip, numpy as np

from __config__.filepath import AUX_DIR

k_range = [0.0, 0.5, 1.0, 1.35]
scale_range = [0.2, 0.5, 0.8, 1.0, 1.2]

SUBDIR = 'meanfield_link'

def process(name, directory=SUBDIR, C_index=9):
    """
    Fit link weights to observed frequencies
    """
    print(name)
    with gzip.open(os.path.join(AUX_DIR, directory, name+'-raw.pkl.gz'), 'rb') as f:
        obs = pickle.load(f)
    
    D = obs.shape[1]
    C = int(name.split('-')[C_index])

    pmi = (np.log(obs) - 2 * np.log(C/D))
    
    # Negative offset for pmi
    for k in k_range:
        print('k', k)
        nonneg = (pmi - k).clip(0)
        
        # Scale pmi values
        for scale in scale_range:
            print('scale', scale)
            link_wei = nonneg * scale
            
            # Calculate entity bias
            # We want the average bias for 'correct' dimensions to be 0, as was used in the meanfield approximation
            # Multiply mean by 4, so that it's the mean of noun-verb connections
            ent_bias = link_wei.mean() * 4 * C + np.log(D/C - 1)
            
            # Save parameters
            full_name = name + '-{}-{}'.format(*[str(x).replace('-','~').replace('.','_') for x in (k, scale)])
            with gzip.open(os.path.join(AUX_DIR, directory, full_name+'.pkl.gz'), 'wb') as f:
                pickle.dump(link_wei, f)
            with gzip.open(os.path.join(AUX_DIR, directory, full_name+'-bias.pkl.gz'), 'wb') as f:
                pickle.dump(ent_bias, f)
    
    print('done')

# Alternatively, use observed frequency of each dimension to fit ent biases,
# and also for the null hypothesis in calculating pmi 
# (for C=40, observed varies between .089 and .103, instead of exactly .1)

if __name__ == '__main__':
    from multiprocessing import Pool
    
    files = []
    for filename in os.listdir(os.path.join(AUX_DIR, SUBDIR)):
        rawname = filename.split('.')[0]
        name, last = rawname.rsplit('-', 1)
        if last == 'raw':
            files.append(name)
    
    with Pool(4) as p:
        p.map(process, files)
