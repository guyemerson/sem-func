import pickle, os, sys
from numpy import arange, empty, inf, random, unravel_index, zeros
from copy import copy
from multiprocessing import Pool, Manager, TimeoutError
from time import sleep, time
from random import shuffle

from trainingsetup import TrainingSetup
from utils import sub_namespace, sub_dict
from __config__.filepath import DATA_DIR, INIT_DIR

def create_particle(p_full, p_agent, p_patient):
    """
    Create a fantasy particle with a given number of
    transitive and intransitive situations
    """
    particle = []
    nid = 0
    for _ in range(p_full):
        particle.extend([(nid, [0,1], [nid+1,nid+2], [], []),
                         (nid+1, [], [], [0], [nid]),
                         (nid+2, [], [], [1], [nid])])
        nid += 3
    for _ in range(p_agent):
        particle.extend([(nid, [0], [nid+1], [], []),
                         (nid+1, [], [], [0], [nid])])
        nid += 2
    for _ in range(p_patient):
        particle.extend([(nid, [1], [nid+1], [], []),
                         (nid+1, [], [], [1], [nid])])
        nid += 2
    return particle


class DataInterface():
    """
    A semantic function model with a training regime and data
    """
    def __init__(self, setup, particle, neg_samples, ent_burnin=0, pred_burnin=0):
        """
        Initialise the data interface
        :param setup: semantic function model with training setup
        :param particle: fantasy particle of the form (nodeid, out_labs, out_ids, in_labs, in_ids), with increasing nodeids 
        :param neg_samples: number of negative pred samples to draw for each node
        :param ent_burnin: (default 0) number of update steps to take for latent entities
        :param pred_burnin: (default 0) number of update steps to take for negative preds
        """
        # Training setup
        self.setup = setup
        self.model = setup.model
        # Negative pred samples
        self.NEG = neg_samples
        # Data
        self.filename = None
        # Fantasy particles
        self.neg_nodes = particle
        self.neg_link_counts = zeros(self.model.L)
        for i, n in enumerate(self.neg_nodes):
            assert i == n[0]
            for label in n[1]:  # Count outgoing links only, and assume we have entire graphs (similarly, only outgoing links are observed)
                self.neg_link_counts[label] += 1
        self.K = len(self.neg_nodes)
        self.neg_ents = random.binomial(1, self.model.C/self.model.D, (self.K, self.model.D))
    
    def load_data(self, data, ent_burnin=0, pred_burnin=0, check=False):
        """
        Load data from a list
        :param data: observed data of the form (nodeid, pred, out_labs, out_ids, in_labs, in_ids), with increasing nodeids
        :param ent_burnin: number of update steps to take for latent entities
        :param pred_burnin: number of update steps to take for negative preds
        """
        # List of nodes
        self.nodes = data
        self.N = len(self.nodes)
        # Optionally, check that nodeids are increasing integers
        if check:
            for i, n in enumerate(self.nodes): assert i == n[0]
        # Latent entities
        self.ents = empty((self.N, self.model.D))
        for i, n in enumerate(self.nodes):
            # TODO use max vec?
            self.ents[i] = self.model.init_vec_from_pred(n[1])  #!# Not currently controlling high and low limits
        for _ in range(ent_burnin):
            self.setup.resample_conditional_batch(self.nodes, self.ents)
        # Negative pred samples
        self.neg_preds = self.model.propose_pred((self.N, self.NEG))
        for _ in range(pred_burnin):
            self.setup.resample_pred_batch(self.nodes, self.ents, self.neg_preds)
    
    def load_file(self, filehandle, ent_burnin=0, pred_burnin=0):
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
                self.setup.train_batch(batch, self.ents, self.neg_preds, self.neg_nodes, self.neg_ents, self.neg_link_counts)
                
            # Print regularly
            if (e+1) % print_every == 0:
                # Print a summary
                self.report(histogram_bins, bias_histogram_bins)
                # Save to file
                if dump_file:
                    with open(dump_file, 'wb') as f:
                        pickle.dump(self.setup, f)

class MultiPredDataInterface(DataInterface):
    """
    Data interface that allows multiple preds for the same node
    """
    def load_data(self, data, ent_burnin=0, pred_burnin=0, check=False):
        """
        Load data from a list
        :param data: observed data of the form (nodeid, preds, out_labs, out_ids, in_labs, in_ids), with increasing nodeids
        :param ent_burnin: number of update steps to take for latent entities
        :param pred_burnin: number of update steps to take for negative preds
        """
        # List of nodes
        self.nodes = data
        self.N = len(self.nodes)
        # Optionally, check that nodeids are increasing integers
        if check:
            for i, n in enumerate(self.nodes): assert i == n[0]
        # Latent entities
        self.ents = empty((self.N, self.model.D))
        for i, n in enumerate(self.nodes):
            # TODO use max vec?
            self.ents[i] = self.model.init_vec_from_pred(n[1])  #!# Not currently controlling high and low limits
        for _ in range(ent_burnin):
            self.setup.resample_conditional_batch(self.nodes, self.ents)
        # Negative pred samples
        self.neg_preds = self.model.propose_pred((self.N, self.NEG))
        for _ in range(pred_burnin):
            self.setup.resample_pred_batch(self.nodes, self.ents, self.neg_preds)
        
        # There are two options here:
        # 1. Multiply the neg pred gradients by the number of positive preds
        # 2. Sample extra negative preds


class Trainer():
    """
    Handle processes during training
    """
    def __init__(self, interface, data_dir, output_name, processes, epochs, minibatch, ent_burnin, pred_burnin):
        """
        Initialise the trainer
        :param interface: DataInterface object
        :param data_dir: directory including data files
        :param output_name: name for output files (without file extension)
        :param processes: number of processes
        :param epochs: number of passes over each data point
        :param minibatch: number of nodes in each minibatch
        :param ent_burnin: number of Metropolis-Hastings steps to take when initialising entities
        :param pred_burnin: number of Metropolis-Hastings steps to take when initialising negative preds
        """
        self.interface = interface
        self.setup = interface.setup
        self.model = interface.model
        self.data_dir = data_dir
        self.output_name = output_name
        self.processes = processes
        self.epochs = epochs
        self.minibatch = minibatch
        self.ent_burnin = ent_burnin
        self.pred_burnin = pred_burnin
        
        # Aux info (outside of setup)
        self.aux_info = sub_namespace(self, ["epochs",
                                             "minibatch",
                                             "processes",
                                             "ent_burnin",
                                             "pred_burnin"])
        self.aux_info["neg_samples"] = self.interface.NEG
        self.aux_info["particle"] = self.interface.neg_nodes
        manager = Manager()  # a multiprocessing.Manager object to manage a shared list of completed files
        self.completed_files = manager.list()
        
        self.training = False
        self.error = None
        
    # Functions for multiprocessing
    
    def train_on_file(self, fname):
        """
        Train on a single file
        (without saving to disk)
        """
        print(fname)
        with open(os.path.join(self.data_dir, fname), 'rb') as f:
            self.interface.load_file(f,
                                     ent_burnin = self.ent_burnin,
                                     pred_burnin = self.pred_burnin)
        self.interface.train(epochs = self.epochs,
                             minibatch = self.minibatch)
        self.completed_files.append(fname)
        
    def start(self, timeout=None, validation=None):
        """
        Begin training
        :param timeout: max number of hours to spend
        """
        # Workers to update the shared weights from queued gradients
        self.setup.start_update_workers()        
        
        # Files to be trained on
        file_name_set = set(os.listdir(self.data_dir))
        file_name_set -= set(self.completed_files)
        if validation:
            file_name_set -= set(validation)
        file_names = list(file_name_set)
        shuffle(file_names)
        print('{} files to process'.format(len(file_names)))
        
        # Process the files with a pool of worker processes
        with Pool(self.processes, self.init) as p:  
            self.training = True
            self.error = None
            initial_time = time()
            p.map_async(self.work, file_names, 1, self.callback, self.error_callback)
            while self.training:
                if self.error:
                    # Re-raise errors from worker processes
                    print('Error during training!')
                    self.kill_queues()  # Kill all update workers, so that the process can exit
                    raise self.error
                elif timeout and time() - initial_time > timeout*3600:
                    raise TimeoutError
                else:
                    # Save regularly during training
                    self.save_and_sleep()
        
        # Once workers are done:
        self.kill_queues()
        while not (all(q.empty() for q in self.setup.link_update_queues) and \
                   all(q.empty() for q in self.setup.pred_update_queues)):
            print('Waiting for updates to finish...')
            self.save_and_sleep()
        print('Training complete')
        self.save()
    
    # The following four functions are passed to Pool.map_async, during training
    
    # Callbacks for Pool.map_async
    # These just set attributes, which will be checked once the instance stops sleeping
    def callback(self, _):
        self.training = False
    def error_callback(self, e):
        self.error = e
    
    # Initialise each worker with the instance's train_on_file method (as a global variable),
    # so that it is not pickled and piped with each file
    def init(self):
        global train_on_file
        train_on_file = self.train_on_file
    # The train_on_file function will be available in each worker
    # This is a static method so that it can be pickled - from Python 3.5, see https://bugs.python.org/issue23611
    @staticmethod
    def work(fname):
        train_on_file(fname)
        # TODO - sometimes one worker hangs...
    
    def kill_queues(self):
        """
        Put None on each queue, to signal that that worker should stop
        """
        for q in self.setup.link_update_queues:
            q.put(None)
        for q in self.setup.pred_update_queues:
            q.put(None)
    
    def save_and_sleep(self, time=60):
        """
        Save, flush stdout, and sleep
        """
        self.save()
        sys.stdout.flush()
        sleep(60)
    
    def save(self):
        """
        Save trained model to disk
        """
        with open(self.output_name+'.pkl', 'wb') as f:
            # Shallow copy of the setup
            crucial = copy(self.setup)
            # Remove queues (which can't be pickled)
            crucial.link_update_queues = [q.qsize() for q in crucial.link_update_queues]
            crucial.pred_update_queues = [q.qsize() for q in crucial.pred_update_queues]
            # Save the file!
            pickle.dump(crucial, f)
        with open(self.output_name+'.aux.pkl', 'wb') as f:
            # Save the aux info, and the list of completed files
            actual_info = copy(self.aux_info)
            actual_info['completed_files'] = self.completed_files._getvalue()
            pickle.dump(actual_info, f)
    
    @staticmethod
    def load(fname, directory=INIT_DIR, data_dir=None, output_name=None, output_dir=None):
        """
        Load trained model from disk
        """
        if output_dir == None:
            output_dir = directory
        if output_name == None:
            output_name = fname
            while os.path.exists(os.path.join(output_dir, output_name)+'.pkl'):
                output_name += '_ctd'
        
        setup, aux_info = TrainingSetup.load(fname, directory)
        
        interface = DataInterface(setup, (),
                                  **sub_dict(aux_info, ['particle',
                                                        'neg_samples']))
        
        if data_dir is None:
            stem = '-'.join(fname.split('-', maxsplit=2)[:2])
            data_dir = os.path.join(DATA_DIR,'{}-nodes'.format(stem))
        
        trainer = Trainer(interface,
                          data_dir = data_dir,
                          output_name = os.path.join(output_dir, output_name),
                          **sub_dict(aux_info, ['processes',
                                                'epochs',
                                                'minibatch',
                                                'ent_burnin',
                                                'pred_burnin']))
        
        trainer.completed_files.extend(aux_info['completed_files'])
        
        return trainer
