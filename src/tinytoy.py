from dmrs.core import PointerNode as Node, Link, DictPointDmrs as Dmrs
import pickle

noun = ['dog','duck', 'pigeon']
verb = ['run','swim','fly','eat','breathe','bark','growl','quack','coo']

I = len(noun)

sent = [[1,0,1],
        [1,1,0],
        [0,1,1],
        [1,1,1],
        [1,1,1],
        [1,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1]]

dmrs = []
n = 0

for i, row in enumerate(sent):
    for j, value in enumerate(row):
        if value:
            nodes = [Node(n, I+i),
                     Node(n+1, j)]
            links = [Link(n, n+1, 0, None)]
            dmrs.append(Dmrs(nodes, links))
            n += 2

with open('../data/tinytoy.pkl','wb') as f:
    pickle.dump(dmrs, f)