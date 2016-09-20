import numpy as np
import pickle
from math import log

with open('/anfs/bigdisc/gete2/wikiwoods/sem-func/bootstrap_link_400.pkl', 'rb') as f:
    link_mat = pickle.load(f)
link_mat_totals = link_mat.sum(axis=1).sum(axis=1)
link_mat /= link_mat_totals.reshape((2,1,1))
link_mat *= (40**2)  # Total observed components per link
link_mat = np.log(link_mat)
link_mat += log((400/40)**2)  # Uniform probability
link_mat.clip(0, out=link_mat)
link_mat.dump('/anfs/bigdisc/gete2/wikiwoods/sem-func/bootstrap_link_log400.pkl')