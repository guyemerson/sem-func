import pickle, gzip, re
from delphin.mrs import simplemrs, dmrx
from dmrs import core

regex = re.compile(r'(_<[^:<> ]*>)')
gpred = re.compile(r'<gpred></gpred>')

graphs = []

# Assign indices to all preds and links
class Index(dict):
    def __init__(self):
        self.n = 0
    
    def find(self, name):
        try:
            return self[name]
        except KeyError:
            self[name] = self.n
            self.n += 1
            return self.n-1

preds = Index()
links = Index()
n_node = 0

with gzip.open('../data/cathbaz/result.gz', 'r') as f:
    for line in f:
        # Take just the MRS from the treebank
        mrs_string = line.decode().rsplit('@', maxsplit=2)[1]
        # Angle brackets in pred names don't work in pydelphin
        x = regex.findall(mrs_string)
        if x:
            print(mrs_string)
            print(x)
            mrs_string = regex.sub('_url', mrs_string)
        # Load the MRS using pydelphin, convert it to XML format, then load it using pydmrs
        mrs = simplemrs.loads_one(mrs_string)
        xml = dmrx.dumps_one(mrs)[11:-12]
        dmrs = core.DictPointDmrs.loads_xml(xml.encode())
        # Renumber nodes and preds
        if n_node >= 10000 and n_node < 10000+len(dmrs):
            for node in dmrs.nodes:
                node.renumber(node.nodeid-10000) # Keep the old values out of the way
        for node in dmrs.nodes:
            # Unique nodeid
            node.renumber(n_node)
            n_node += 1
            # Pred index
            node.pred = preds.find(node.pred)
        # Renumber links
        for node in dmrs.nodes:
            outlinks = {core.Link(l.start, l.end, links.find(l.label), None) for l in node.outgoing}
            inlinks = {core.Link(l.start, l.end, links.find(l.label), None) for l in node.incoming}
            dmrs.outgoing[node.nodeid] = outlinks
            dmrs.incoming[node.nodeid] = inlinks
        graphs.append(dmrs)

with open('../data/cathbaz/dmrs.pkl', 'wb') as f:
    pickle.dump((graphs, dict(preds), dict(links)), f)