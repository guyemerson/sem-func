import pickle, gzip, re
from delphin.mrs import simplemrs, dmrx
from dmrs import core

regex = re.compile(r'(_<[^:<> ]*>)')
gpred = re.compile(r'<gpred></gpred>')

graphs = []

with gzip.open('../data/cathbaz/mrs.gz', 'r') as f:
    for line in f:
        mrs_string = line.decode().rsplit('@', maxsplit=2)[1]
        x = regex.findall(mrs_string)
        if x:
            print(mrs_string)
            print(x)
            mrs_string = regex.sub('_url', mrs_string)
        empty = gpred.findall(mrs_string)
        if empty:
            print(empty)
        mrs = simplemrs.loads_one(mrs_string)
        xml = dmrx.dumps_one(mrs)[11:-12]
        dmrs = core.DictPointDmrs.loads_xml(xml.encode())
        graphs.append(dmrs)

with open('../data/cathbaz/dmrs.pkl', 'wb') as f:
    pickle.dump(graphs, f)