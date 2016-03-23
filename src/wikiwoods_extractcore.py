import sys, os, gzip, pickle
from xml.etree.ElementTree import ParseError
from traceback import print_tb

from pydmrs.components import RealPred, GPred
from pydmrs.core import ListDmrs as Dmrs

def find_sit(dmrs, node):
    """
    Find if a node representations a situation
    :param dmrs: a Dmrs object
    :param node: a Node object
    :return: (verb, agent, patient), realpred_only
        or if not found: None, None
    """
    # Only consider verbal nodes
    if not isinstance(node.pred, RealPred) or node.pred.pos != 'v':
        return None, None
    # Output of the form (verb, agent, patient)
    output = [node.pred, None, None]
    # Record if arguments are RealPreds
    realpred_only = True
    # Look for both ARG1 and ARG2
    for i in (1,2):
        try:  # See if the argument is there
            arglink = dmrs.get_out(node.nodeid, 'ARG'+str(i)).pop()
        except KeyError:
            continue
        # Get argument's pred
        end = dmrs[arglink.end]
        pred = end.pred
        # Deal with different pred types
        if type(pred) == RealPred:
            # Ignore coordinations
            if pred.pos == 'c':
                continue
            else:
                output[i] = pred
        else:
            # Note that this pred is not a RealPred
            realpred_only = False
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
            elif pred == GPred('named'):
                output[i] = end.carg
            else:
                output[i] = pred
    # Check if an argument was found
    if output[1] or output[2]:
        return output, realpred_only
    else:
        return None, None

def extract(xmlstring, sits, extra_sits, filename):
    """
    Extract situations from a DMRS in XML form
    :param xmlstring: the input XML
    :param sits: the list of situations to append to
    :param extra_sits: the list of extra situations (including GPreds) to append to
    :param filename: the filename to log errors to
    """
    # Replace characters that cause parse errors
    xmlstring.replace(b'<', b'&lt;')
    xmlstring.replace(b'&', b'&amp;')
    try:
        dmrs = Dmrs.loads_xml(xmlstring)
    except ParseError as e:  # badly formed XML
        print("ParseError!")
        with open('wikiwoods_extractcore.log', 'a') as f:
            f.write(filename + ':\n' + xmlstring.decode() + '\n' + str(e) + '\n')
        return None
    # Look for situations
    for n in dmrs.iter_nodes():
        situation, realpred_only = find_sit(dmrs, n)
        if situation:
            if realpred_only:
                sits.append(situation)
            else:
                extra_sits.append(situation)

# Directory of DMRSs, and directory to save triples
SOURCE = '/usr/groups/corpora/wikiwoods-1212-tmp/dmrs/'
TARGET = '/anfs/bigdisc/gete2/wikiwoods/core'
EXTRA = '/anfs/bigdisc/gete2/wikiwoods/core-extra'

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
            extra_sits = []
            # Each xml will span multiple lines,
            # separated by an empty line
            f.readline() # The first line is blank, for some reason
            xml = b''
            for line in f:
                # Keep adding new lines until we get a blank line
                if line != b'\n':
                    xml += line
                else:  # Once we've found a blank line, extract the DMRS
                    extract(xml, situations, extra_sits, filename)
                    # Reset the xml string
                    xml = b''
            # If the file does not end with a blank line:
            if xml != b'':
                extract(xml, situations, extra_sits, filename)
        # Save the triples in TARGET
        with open(os.path.join(TARGET, newname), 'wb') as f:
            pickle.dump(situations, f)
        with open(os.path.join(EXTRA, newname), 'wb') as f:
            pickle.dump(extra_sits, f)
    except:
        print("Error!")
        with open('wikiwoods_extractcore.log', 'a') as f:
            f.write(filename+'\n')
            _, error, trace = sys.exc_info()
            f.write(str(error)+'\n')
            print_tb(trace, file=f)
            f.write('\n\n')
        continue
