import pickle
from numpy import arange, empty, inf, random, unravel_index

def create_particle(p_full, p_agent, p_patient):
    """
    Create a fantasy particle with a given number of
    transitive and intransitive situations
    """
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
    for _ in range(p_patient):
        particle.extend([(nid, [1], [nid+1], (), ()),
                         (nid+1, (), (), [1], [nid])])
        nid += 2
    return particle

class DirectTrainer():
    """
    A semantic function model with a training regime and data
    """
    def __init__(self, setup, data, particle, neg_samples):
        """
        Initialise the trainer
        :param setup: semantic function model with training setup
        :param data: observed data of the form (nodeid, pred, out_labs, out_ids, in_labs, in_ids), with increasing nodeids
        :param particle: fantasy particle of the form (nodeid, out_labs, out_ids, in_labs, in_ids), with increasing nodeids 
        :param neg_samples: number of negative pred samples to draw for each node
        """
        # Training setup
        self.setup = setup
        self.model = setup.model
        # Negative pred samples
        self.NEG = neg_samples
        # Data
        self.filename = None
        self.load_data(data)
        # Fantasy particles
        self.neg_nodes = particle
        for i, n in enumerate(self.neg_nodes): assert i == n[0]
        self.K = len(self.neg_nodes)
        self.neg_ents = random.binomial(1, self.model.C/self.model.D, (self.K, self.model.D))
    
    def load_data(self, data, ent_burnin, pred_burnin):
        """
        Load data from a list
        :param data: observed data of the form (nodeid, pred, out_labs, out_ids, in_labs, in_ids), with increasing nodeids
        :param ent_burnin: number of update steps to take for latent entities
        :param pred_burnin: number of update steps to take for negative preds
        """
        # Dicts for graphs, nodes, and pred frequencies
        self.nodes = data
        for i, n in enumerate(self.nodes): assert i == n[0]
        self.N = len(self.nodes)
        # Latent entities
        self.ents = empty((self.N, self.model.D))
        for i, n in enumerate(self.nodes):
            self.ents[i] = self.model.init_vec_from_pred(n[1])
        for _ in range(ent_burnin):
            self.setup.resample_conditional_batch(self.nodes, self.ents)
        # Negative pred samples
        self.neg_preds = empty((self.N, self.NEG), dtype=int)
        for n in self.nodes:
            self.neg_preds[n[0], :] = n[1]  # Initialise all pred samples as the nodes' preds
        for _ in range(pred_burnin):
            self.setup.resample_pred_batch(self.nodes, self.ents, self.neg_preds)
    
    def load_file(self, filehandle, ent_burnin, pred_burnin):
        """
        Load data from a file
        :param filehandle: pickled data
        :param ent_burnin: number of update steps to take for latent entities
        :param pred_burnin: number of update steps to take for negative preds
        """
        data = pickle.load(filehandle)
        self.load_data(data, ent_burnin, pred_burnin)
        self.filename = filehandle.name
        self.epochs = 0
    
    def report(self, histogram_bins, bias_histogram_bins, num_preds=5):
        """
        Print a summary of the current state of training the model
        :param histogram_bins: edges of bins for non-bias weights (0 and inf will be added)
        :param bias_histogram_bins: edges of bins for bias weights (0 and inf will be added)
        :param num_preds: number of preds to print the nearest neighbours of
        """
        # Get histogram
        histo, histo_bias = self.model.get_all_histograms(histogram_bins, bias_histogram_bins)
        # Print to console
        print()
        print('File {} epoch {} complete!'.format(self.filename, self.epochs))
        print('Weight histogram (link, then pred):')
        print(histo)
        print('Bias histogram (pred):')
        print(histo_bias)
        print('max link weights:')
        for m in self.model.link_weights:
            print('\t', m.max())
        print('max global pred weights:')
        for m in self.model.pred_global_weights:
            print('\t', m.max())
        print('max local pred weights:')
        for m in self.model.pred_local_weights:
            i_max = unravel_index(m.argmax(), m.shape)
            print('\t', m[i_max], '\t', self.model.pred_name[i_max[0]])
        print('avg data background E:', self.setup.graph_background_energy(self.nodes, self.ents) / self.N)
        print('avg part background E:', self.setup.graph_background_energy(self.neg_nodes, self.neg_ents) / self.K)  # Check for new samples?
        print('avg data pred t:', sum(self.model.prob(self.ents[n[0]], n[1]) for n in self.nodes) / self.N)
        print('avg part pred t:', sum(self.model.prob(self.ents[n[0]], p) for n in self.nodes for p in self.neg_preds[n[0]]) / self.N / self.NEG)  # Just check different preds?
        #print('closest preds:')
        #for p, q in self.model.closest_pairs(12, 'parameters'):
        #    print(p,q)
        print('nearest neighbours:')
        # Get frequent preds
        if not hasattr(self, 'pred_list'):
            self.pred_list = list(self.model.freq.argpartition(tuple(range(-1,-1-num_preds,-1)))[-num_preds:])
        # Get the first few preds in the current file
        self.pred_list[num_preds:] = [n[1] for n in self.nodes[:num_preds]]
        nearest = self.model.closest_preds(self.pred_list, 3)
        for i, p in enumerate(self.pred_list):
            if nearest[i] is not None:
                neighbours = ', '.join(self.model.pred_name[x] for x in nearest[i])
            else:
                neighbours = ''
            print('{}: {}'.format(self.model.pred_name[p], neighbours))
    
    def train(self, epochs, minibatch, print_every=inf, histogram_bins=(0.05,0.2,1), bias_histogram_bins=(4,5,6,10), dump_file=None):
        """
        Train the model on the data
        :param epochs: number of passes over the data
        :param minibatch: size of a minibatch (as a number of graphs)
        :param print_every: how many epochs should pass before printing
            (default: don't print)
        :param histogram_bins: edges of bins to summarise distribution of weights
            (default: 0.05, 0.2, 1)
        :param bias_histogram_bins: edges of bins to summarise distribution of biases
            (default: 4, 5, 6, 10)
        :param dump_file: (optional) file to save the trained model (dumps after printing)
        """
        # Indices of nodes, to be randomised
        indices = arange(self.N)
        for e in range(epochs):
            # Record that another epoch has passed
            self.epochs += 1
            # Randomise batches
            # (At the moment, just one batch of particles)
            random.shuffle(indices)
            # Take batches
            for i in range(0, self.N, minibatch):
                # Get the nodes for this batch
                batch = [self.nodes[i] for i in indices[i : i+minibatch]]
                # Train on this batch
                self.setup.train_batch(batch, self.ents, self.neg_preds, self.neg_nodes, self.neg_ents)
                
            # Print regularly
            if (e+1) % print_every == 0:
                # Print a summary
                self.report(histogram_bins, bias_histogram_bins)
                # Save to file
                if dump_file:
                    with open(dump_file, 'wb') as f:
                        pickle.dump(self.setup, f)