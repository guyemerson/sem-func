from pydmrs.core import PointerNode as Node, Link, DictPointDmrs as Dmrs
import pickle

noun = ['dog','cat','mouse']
verb = ['chase']

I = len(noun)

sent = [[0,1,0],
        [0,0,1],
        [0,0,0]]

dmrs = []
n = 0

for _ in range(10):
    for i, row in enumerate(sent):
        for j, value in enumerate(row):
            if value:
                nodes = [Node(n, I),
                         Node(n+1, i),
                         Node(n+2, j)]
                links = [Link(n, n+1, 0, None),
                         Link(n, n+2, 1, None)]
                dmrs.append(Dmrs(nodes, links))
                n += 3

with open('../data/trantoy.pkl','wb') as f:
    pickle.dump(dmrs, f) 

# Would like (person feed mouse rice) => (mouse eat rice)
# Also (X buy from Y) => (Y sell to X)
