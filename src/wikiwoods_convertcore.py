import os, pickle
from multiprocessing import Pool  # @UnresolvedImport

from pydmrs.core import RealPred

DATA = '/anfs/bigdisc/gete2/wikiwoods/core-10000'
VOCAB = '/anfs/bigdisc/gete2/wikiwoods/core-5-vocab.pkl'
OUTPUT = DATA + '-nodes'

PROC = 80

if not os.path.exists(OUTPUT):
    os.mkdir(OUTPUT)

with open(VOCAB, 'rb') as f:
    vocab_list = pickle.load(f)

pred_index = {RealPred.from_string(p):i for i,p in enumerate(vocab_list)}

def convert_triple(triple, nid):
    """
    Convert an SVO triple of RealPreds to a form suitable for training
    :param triple: RealPred triple
    :param nid: next free nodeid
    :return: list of nodes, new next free nodeid
    """
    verb, agent, patient = triple
    verb_i = pred_index[verb]
    agent_i = pred_index.get(agent, None)
    patient_i = pred_index.get(patient, None) 
    output = []
    verb_labs = []
    verb_ids = []
    output.append((nid, verb_i, verb_labs, verb_ids, (), ()))
    new_id = nid+1
    if agent:
        output.append((new_id, agent_i, (), (), [0], [nid]))
        verb_labs.append(0)
        verb_ids.append(new_id)
        new_id += 1
    if patient:
        output.append((new_id, patient_i, (), (), [1], [nid]))
        verb_labs.append(1)
        verb_ids.append(new_id)
        new_id += 1
    return output, new_id

def process_file(fname):
    """
    Process all the triples in a file
    :param fname: filename
    """
    print(fname)
    with open(os.path.join(DATA, fname), 'rb') as f:
        trips = pickle.load(f)
    nodes = []
    nid = 0
    for triple in trips:
        packaged, nid = convert_triple(triple, nid)
        nodes.extend(packaged)
    with open(os.path.join(OUTPUT, fname), 'wb') as f:
        pickle.dump(nodes, f)

# Process all files in DATA
with Pool(PROC) as p:
    p.map(process_file, os.listdir(DATA))