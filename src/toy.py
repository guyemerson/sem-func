from dmrs.core import PointerNode as Node, Link, DictPointDmrs as Dmrs
import pickle

noun = ['dog','cat','mouse','person','animal','food','rice','meat']
intr = ['sleep','run','bark','miao','squeak','talk']
tran = ['see','eat','chase','cook']
ditr = ['give']

I = len(noun)
T = I + len(intr)
D = T + len(tran)

intr_sent = [[1,1,1,1,1,0,0,0],
             [1,1,1,1,1,0,0,0],
             [1,0,0,0,1,0,0,0],
             [0,1,0,0,1,0,0,0],
             [0,0,1,0,1,0,0,0],
             [0,0,0,1,0,0,0,0]]

tran_sent = [[[1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0]],
             [[0,0,0,0,1,1,1,1],
              [0,0,1,0,1,1,0,1],
              [0,0,0,0,0,1,1,0],
              [1,0,1,0,1,1,1,1],
              [1,1,1,1,1,1,1,1],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0]],
             [[1,1,0,1,1,0,0,0],
              [0,0,1,0,1,0,0,0],
              [0,0,0,0,0,0,0,0],
              [1,1,1,1,1,0,0,0],
              [1,1,1,1,1,0,0,0],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0]],
             [[0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],
              [1,0,1,0,1,1,1,1],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0]]]

zero = [[0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0]]

ditr_sent = [[zero,
              zero,
              zero,
              [[0,0,0,1,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,0,0,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [1,1,1,1,1,0,0,0],
               [1,0,1,1,1,0,0,0],
               [1,1,0,1,1,0,0,0]],
              zero,
              zero,
              zero,
              zero]]

intr_dmrs = []
tran_dmrs = []
ditr_dmrs = []
n = 0

for i, row in enumerate(intr_sent):
    for j, value in enumerate(row):
        if value:
            nodes = [Node(n, I+i),
                     Node(n+1, j)]
            links = [Link(n, n+1, 0, None)]
            intr_dmrs.append(Dmrs(nodes, links))
            n += 2

for i, mat in enumerate(tran_sent):
    for j, row in enumerate(mat):
        for k, value in enumerate(row):
            if value:
                nodes = [Node(n, T+i),
                         Node(n+1, j),
                         Node(n+2, k)]
                links = [Link(n, n+1, 0, None),
                         Link(n, n+2, 1, None)]
                tran_dmrs.append(Dmrs(nodes, links))
                n += 3

for i, ten in enumerate(ditr_sent):
    for j, mat in enumerate(ten):
        for k, row in enumerate(mat):
            for l, value in enumerate(row):
                if value:
                    nodes = [Node(n, D+i),
                             Node(n+1, j),
                             Node(n+2, k),
                             Node(n+3, l)]
                    links = [Link(n, n+1, 0, None),
                             Link(n, n+2, 1, None),
                             Link(n, n+3, 2, None)]
                    ditr_dmrs.append(Dmrs(nodes, links))
                    n += 4

with open('../data/toy.pkl','wb') as f:
    pickle.dump((intr_dmrs, tran_dmrs, ditr_dmrs), f) 

# Would like (person feed mouse rice) => (mouse eat rice)