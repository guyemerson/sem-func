import sys, os, pickle, numpy
from collections import Counter

from model import DirectTrainingSetup, DirectTrainer, \
    SemFuncModel_IndependentPreds, SemFuncModel_FactorisedPreds

numpy.set_printoptions(precision=3, suppress=True, threshold=numpy.nan)

DATA = '/anfs/bigdisc/gete2/wikiwoods/core'

with open(os.path.join(DATA,'00390.pkl'), 'rb') as f:
    data = pickle.load(f)

pred_count = Counter(p for x in data for p in x if p)
preds = sorted(pred_count.keys(), key=str)
links = ('ARG1','ARG2')
pred_freq = [pred_count[p] for p in preds]

model = SemFuncModel_FactorisedPreds(preds, links, pred_freq,
                                      dims = 50,
                                      card = 5,
                                      embed_dims = 20, 
                                      init_bias = 5,
                                      init_card = 8,
                                      init_range = 1)

setup = DirectTrainingSetup(model,
                            rate = 0.01,
                            rate_ratio = 1,
                            l2 = 0.001,
                            l2_ratio = 100,
                            l1 = 0.000001,
                            l1_ratio = 1,
                            ent_steps = 3,
                            pred_steps = 2)

pred_index = {p:i for i,p in enumerate(preds)}

n_agent = 0
n_patient = 0

def convert_triple(triple, nid):
    global n_agent, n_patient
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
        n_agent += 1
    if patient:
        output.append((new_id, patient_i, (), (), [1], [nid]))
        verb_labs.append(1)
        verb_ids.append(new_id)
        new_id += 1
        n_patient += 1
    return output, new_id

nodes = []
nid = 0
for triple in data:
    packaged, nid = convert_triple(triple, nid)
    nodes.extend(packaged)

full = n_agent+n_patient-len(data)
agent_only = len(data) - n_patient
patient_only = len(data) - n_agent
total = full + agent_only + patient_only

N_PART = 5

p_full = round(full * N_PART / total)
p_agent = round(agent_only * N_PART / total)
p_patient = round(patient_only * N_PART / total)

print(p_full, p_agent, p_patient)

particle = []
nid = 0
for _ in range(p_full):
    particle.extend([(nid, [0,1], [nid+1,nid+2], (), ()),
                     (nid+1, (), (), [0], [nid]),
                     (nid+2, (), (), [1], [nid])])
    nid += 3
for _ in range(p_agent):
    particle.extend([(nid, [0], [nid+1], (), ()),
                     (nid+1, (), (), [0], [nid])])
    nid += 2
for _ in range(p_agent):
    particle.extend([(nid, [1], [nid+1], (), ()),
                     (nid+1, (), (), [1], [nid])])
    nid += 2

trainer = DirectTrainer(setup, nodes, particle,
                      neg_samples = 5)

print("Set up complete, beginning training...")
sys.stdout.flush()


trainer.train(epochs = 100,
              minibatch = 20,
              print_every = 1)

"""
import cProfile
cProfile.runctx('trainer.train(1, 20, 1)',globals(),locals())
"""