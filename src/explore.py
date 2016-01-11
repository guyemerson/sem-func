import pickle
from model import SemFuncModel

with open('../data/out.pkl', 'rb') as f:
    model = pickle.load(f)