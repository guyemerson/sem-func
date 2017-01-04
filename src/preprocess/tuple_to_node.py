import os, pickle
from random import shuffle
from math import floor
from multiprocessing import Pool

# Load data

print('loading')

with open('/anfs/bigdisc/gete2/wikiwoods/multicore-5-count_tuple.pkl', 'rb') as f:
    count = pickle.load(f)

# Explicit list of graphs, with repeats, in a random order

print('shuffling')

data = []
for graph, n in count.items():
    data.extend([graph]*n)
shuffle(data)

# Randomly split into chunks

print('chunking')

N = len(data)
K = 10000
splits = [floor(i*N/K) for i in range(K)] + [N]
chunks = [data[splits[i]:splits[i+1]] for i in range(K)]
del data

# Convert data to the necessary form

def add_graph(nodes, graph):
    """
    Convert data a form suitable for training, adding it to a list
    :param nodes: list of nodes 
    :param graph: SVO triple, or pair linked by _be_v_id
    :return: list of nodes, new next free nodeid
    """
    next_id = len(nodes)
    if len(graph) == 2:
        # two preds linked by _be_v_id
        nodes.append((next_id, list(graph), [], [], [], []))
    else:
        # a verb and up to two arguments
        verb, agent, patient = graph
        output = []  # nodes to add to the list 
        verb_labs = []  # verb arguments, yet to be populated
        verb_ids = []
        output.append((next_id, [verb], verb_labs, verb_ids, [], []))
        new_id = next_id + 1
        if agent is not None:
            output.append((new_id, [agent], [], [], [0], [next_id]))
            verb_labs.append(0)
            verb_ids.append(new_id)
            new_id += 1
        if patient is not None:
            output.append((new_id, [patient], [], [], [1], [next_id]))
            verb_labs.append(1)
            verb_ids.append(new_id)
            new_id += 1
        nodes.extend(output)

# Convert and save

print('converting')

DIR = '/anfs/bigdisc/gete2/wikiwoods/multicore-5-nodes-tmp'

def process(i, chunk):
    """
    Convert the graphs in a chunk and save to disk
    """
    nodes = []
    for g in chunk:
        add_graph(nodes, g)
    with open(os.path.join(DIR, '{}.pkl'.format(str(i).zfill(4))), 'wb') as f:
        pickle.dump(nodes, f)
    print(i)

with Pool(30) as p:
    p.starmap(process, enumerate(chunks))