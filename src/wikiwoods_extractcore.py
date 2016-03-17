import sys, os, gzip, pickle
from xml.etree.ElementTree import ParseError

from pydmrs.components import RealPred, GPred
from pydmrs.core import ListDmrs as Dmrs


def extract(xmlstring, sits, filename):
    """Extract situations from a DMRS in XML form"""
    xmlstring.replace(b'<', b'&lt;')
    xmlstring.replace(b'&', b'&amp;')
    try:
        dmrs = Dmrs.loads_xml(xmlstring)
    except ParseError as e:  # badly formed XML
        print("ParseError!")
        with open('wikiwoods_extractcore.log', 'a') as f:
            f.write(filename + ':\n' + xmlstring.decode() + '\n' + str(e) + '\n')
        return None
    # Look for verb nodes
    for n in dmrs.iter_nodes():
        if type(n.pred) == RealPred and n.pred.pos == 'v':
            # Record the verb and its ARG1 and ARG2
            output = [n.pred, None, None]
            for i in (1,2):
                try:  # See if the argument is there
                    arglink = dmrs.get_out(n.nodeid, 'ARG'+str(i)).pop()
                except KeyError:
                    continue
                # Get the node and the pred
                end = dmrs[arglink.end]
                pred = end.pred
                if type(pred) == RealPred:
                    # Ignore coordinations
                    if pred.pos == 'c':
                        continue
                    else:
                        output[i] = pred
                else:
                    # Ignore coordinations
                    if pred == GPred('implicit_conj'):
                        continue
                    # Record information about pronouns
                    elif pred == GPred('pron'):
                        pronstring = end.sortinfo['pers']
                        try:
                            pronstring += end.sortinfo['num']
                        except TypeError:  # info is None
                            pass
                        try:
                            pronstring += end.sortinfo['gend']
                        except TypeError:  # info is None
                            pass
                        output[i] = pronstring
                    else:
                        output[i] = pred
            # Only record the verb if one of the arguments was present
            if output[1] or output[2]:
                sits.append(output)

# Directory of DMRSs, and directory to save triples
SOURCE = '/usr/groups/corpora/wikiwoods-1212-tmp/dmrs/'
TARGET = '/anfs/bigdisc/gete2/wikiwoods/core'

# Process each file in SOURCE
for filename in sorted(os.listdir(SOURCE)):
    newname = os.path.splitext(filename)[0]+'.pkl'
    if os.path.exists(os.path.join(TARGET, newname)):
        print('skipping '+filename)
        continue
    try:
        with gzip.open(os.path.join(SOURCE, filename),'rb') as f:
            print(filename)
            # List of situation triples
            situations = []
            # Each xml will span multiple lines,
            # separated by an empty line
            f.readline() # The first line is blank, for some reason
            xml = b''
            for line in f:
                # Keep adding new lines until we get a blank line
                if line != b'\n':
                    xml += line
                else:  # Once we've found a blank line, extract the DMRS
                    extract(xml, situations, filename)
                    xml = b''
            # If the file does not end with a blank line:
            if xml != b'':
                extract(xml, situations, filename)
        # Save the triples in TARGET
        with open(os.path.join(TARGET, newname), 'wb') as f:
            pickle.dump(situations, f)
    except:
        print("Error!")
        with open('wikiwoods_extractcore.log', 'a') as f:
            f.write(filename+'\n')
            f.write(str(sys.exc_info())+'\n\n')
        continue
