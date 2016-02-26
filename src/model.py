from numpy import array, random, tensordot, dot, zeros, outer, arange, absolute, sign, minimum, amax, convolve
from scipy.special import expit
from scipy.spatial.distance import cosine
from math import sqrt, exp
from collections import Counter
import pickle
from pydmrs.core import PointerNode, DictPointDmrs

class SemFuncModel():
    def __init__(self, corpus, neg_graphs, dims, card, rate, rate_ratio, l2, l2_ratio, l1, l1_ratio, bias, init_range, minibatch, print_every, neg_samples):
        """
        Corpus and neg_graphs should each have distinct nodeids for all nodes
        """
        # Hyperparameters
        self.rate_link = rate / sqrt(rate_ratio)
        self.rate_pred = rate * sqrt(rate_ratio)
        self.L2_link = 1 - 2 * self.rate_link * l2 / sqrt(l2_ratio)
        self.L2_pred = 1 - 2 * self.rate_pred * l2 * sqrt(l2_ratio)
        self.L1_link = self.rate_link * l1 / sqrt(l1_ratio)
        self.L1_pred = self.rate_pred * l1 * sqrt(l1_ratio)
        self.minibatch = minibatch
        self.bias = bias
        # Print options
        self.print_every = print_every
        # Dicts for graphs, nodes, and pred frequencies
        self.graphs = dict(enumerate(corpus))
        self.nodes = {n.nodeid:n for x in corpus for n in x.iter_nodes()}
        self.freq = Counter(n.pred for n in self.nodes.values())
        # Dimensions of matrices
        self.D = dims
        self.N = len(self.nodes)
        self.V = 1 + max(n.pred for n in self.nodes.values())
        self.L = 1 + max(x.rargname for n in self.nodes.values() for x in n.outgoing)
        # Latent entities, link weights, and pred weights
        self.C = card
        self.Crange = arange(card+1)
        self.link_wei = zeros((self.L, self.D, self.D))  # link, from, to
        self.pred_wei = random.uniform(0, init_range, (self.V, self.D))
        self.ents = zeros((self.N, self.D))
        self.resample(self.nodes.values(), self.ents)  # Initialise ents just using preds (link weights set to zero)
        #self.link_wei = random.uniform(0, init_range, (self.L, self.D, self.D))
        self.link_sumsq = zeros((self.L, self.D, self.D))
        self.pred_sumsq = zeros((self.V, self.D))
        self.calc_av_pred()
        # Particles for negative samples
        self.neg_graphs = dict(enumerate(neg_graphs))
        self.neg_nodes = {n.nodeid:n for x in neg_graphs for n in x.iter_nodes()}
        self.K = len(self.neg_nodes)
        self.neg_ents = random.binomial(1, 0.5, (self.K, self.D))
        # Weight for negative samples
        pos_links = sum(len(n.outgoing) for n in self.nodes.values())
        neg_links = sum(len(n.outgoing) for n in self.neg_nodes.values())
        self.neg_link_weight = pos_links / neg_links
        #self.neg_pred_weight = len(self.nodes) / len(self.neg_nodes)
        # Negative pred samples
        self.NEG = neg_samples
        self.neg_preds = {n.nodeid:[n.pred]*neg_samples for n in self.nodes.values()}
        
        self.approx = []
    
    # Training functions
    
    def calc_av_pred(self):
        self.av_pred = sum(self.freq[i] * self.pred_wei[i] for i in range(self.V)) / self.N  # average predicate
    
    def resample(self, nodes, ents, pred=True):
        for n in nodes:
            # Find the negative energy for each dimension
            if pred:
                negenergy = array(self.pred_wei[n.pred, :], copy=True)
            else:
                negenergy = zeros(self.D)

            for link in n.outgoing:
                negenergy += tensordot(self.link_wei[link.rargname, :, :],
                                       ents[link.end, :],
                                       (1,0))
            for link in n.incoming:
                negenergy += tensordot(self.link_wei[link.rargname, :, :],
                                       ents[link.start, :],
                                       (0,0))
            # Expit gives the probability without controlling sparsity
            prob = expit(negenergy) # Warning! If the negenergy is above 710, expit returns nan
            minp = 1 - prob
            # Pass messages up
            intermed = [array([minp[0], prob[0]])]
            for i in range(1,self.D-1):
                message = convolve(intermed[-1], [minp[i], prob[i]])[:self.C+1]
                intermed.append(message)
            # Sample total
            #probtotal = intermed[-1]
            #probtotal /= sum(probtotal)
            #print(probtotal)
            #aux = random.choice(self.Crange, p=probtotal)
            #print(aux)
            # Fix total
            aux = self.C
            # Iteratively sample
            nid = n.nodeid
            for i in range(self.D-1, 0, -1):
                # aux holds the total number of 1s remaining
                if aux == i+1:
                    for j in range(i+1):
                        ents[nid, j] = 1
                    break
                elif aux > 0:
                    #print(intermed[i-1])
                    ein = prob[i] * intermed[i-1][aux-1]
                    aus = minp[i] * intermed[i-1][aux]
                    on = ein/(ein+aus)
                    new = random.binomial(1, on)
                    #print(new)
                    ents[nid, i] = new
                    aux -= new
                else:
                    ents[nid, i] = 0
            ents[nid, 0] = aux
            #print(ents[nid])
    
    def sample_latent(self):
        self.resample(self.nodes.values(), self.ents)
    
    def sample_latent_batch(self, nodes):
        self.resample(nodes, self.ents)
    
    def sample_particle(self):
        self.resample(self.neg_nodes.values(), self.neg_ents, pred=False)
    
    def sample_particle_batch(self, nodes):
        self.resample(nodes, self.neg_ents, pred=False)
    
    def resample_metro(self, nodes, ents):
        # uniformly pick an on and an off unit to switch
        # calculate ratio of exp(-E)*pred(x) for x and x'
        # calculate probability of average predicate for x and x'
        # accept or reject x' based on ratio 
        for n in nodes:
            # Pick an on and an off unit to switch
            old_jth = random.randint(self.C)
            new_jth = random.randint(self.D-self.C)
            old_ent = ents[n.nodeid]
            on = 0
            for i, val in enumerate(old_ent):
                if val:
                    if on == old_jth:
                        old_i = i
                        break
                    else:
                        on += 1
            off = 0
            for i, val in enumerate(old_ent):
                if not val:
                    if off == new_jth:
                        new_i = i
                        break
                    else:
                        off += 1
            new_ent = array(old_ent)
            new_ent[old_i] = 0
            new_ent[new_i] = 1
            
            # Calculate Metropolis-Hastings ratio
            # First, background energy of entities:
            negenergy = 0
            for link in n.outgoing:
                negenergy += dot(self.link_wei[link.rargname, new_i, :],
                                 ents[link.end, :])
                negenergy -= dot(self.link_wei[link.rargname, old_i, :],
                                 ents[link.end, :])
            for link in n.incoming:
                negenergy += dot(self.link_wei[link.rargname, :, new_i],
                                 ents[link.start, :])
                negenergy -= dot(self.link_wei[link.rargname, :, old_i],
                                 ents[link.start, :])
            ratio = exp(negenergy)
            # Next, probability of the predicate being applicable:
            ratio *= self.prob(new_ent, n.pred)
            ratio /= self.prob(old_ent, n.pred)
            # Finally, weighted number of other predicates that are true:
            
            exact_ratio = ratio
            
            # Use an approximation...
            ratio *= exp(0.5*(self.av_pred[old_i] - self.av_pred[new_i]))
            
            # Exact... slow!
            exact_ratio /= sum(self.freq[i] * self.prob(new_ent, i) for i in range(self.V))
            exact_ratio *= sum(self.freq[i] * self.prob(old_ent, i) for i in range(self.V))
            
            self.approx.append((ratio, exact_ratio))
            
            # Accept or reject the new entity
            if ratio > 1:
                switch = True
            else:
                switch = random.binomial(1, ratio)
            if switch:
                ents[n.nodeid] = new_ent
        
    def contrast(self):
        # Corpus
        link_obs = zeros((self.L, self.D, self.D))
        link_neg = zeros((self.L, self.D, self.D))
        pred_obs = zeros((self.V, self.D))
        pred_neg = zeros((self.V, self.D))
        for i,n in self.nodes.items():
            # Reinforce observed links
            for link in n.outgoing:
                link_obs[link.rargname, :,:] += outer(self.ents[i,:], self.ents[link.end, :])
            # Reinforce observed preds
            pred_obs[n.pred, :] += self.ents[i,:]
            # Negatively sample another entity
            m = random.randint(self.N)
            pred_neg[n.pred, :] += self.ents[m,:]
        # Particle
        for i,n in self.neg_nodes.items():
            for link in n.outgoing:
                link_neg[link.rargname, :,:] += outer(self.neg_ents[i,:], self.neg_ents[link.end, :])
        # Return steps for link weights and pred weights
        return (link_obs - self.neg_link_weight * link_neg,
                pred_obs - pred_neg) #self.neg_pred_weight * pred_neg)
    
    def observe_links(self, nodes, ents):
        link_obs = zeros((self.L, self.D, self.D))
        for n in nodes:
            for link in n.outgoing:
                link_obs[link.rargname, :,:] += outer(ents[n.nodeid, :], ents[link.end, :])
        return link_obs
    
    def observe_preds_old(self, nodes):
        pred_obs = zeros((self.V, self.D))
        for n in nodes:
            pred_obs[n.pred, :] += self.ents[n.nodeid, :]
        return pred_obs
    
    def neg_sample_preds(self, nodes):
        pred_neg = zeros((self.V, self.D))
        for n in nodes:
            m = random.randint(self.N)
            pred_neg[n.pred, :] += self.ents[m,:]
        return pred_neg
    
    def observe_preds_fancy(self, nodes):
        pred_grad = zeros((self.V, self.D))
        for n in nodes:
            m = random.randint(self.N)
            pos = self.ents[n.nodeid, :]
            neg = self.ents[m, :]
            p_pos = self.prob(pos, n.pred)
            p_neg = self.prob(neg, self.nodes[m].pred)
            factor = (1 - p_pos*(1-p_neg))
            diff = pos - neg
            pred_grad[n.pred, :] += factor * diff
        return pred_grad
        
        # The negative samples for preds need to be chosen more carefully
    
    def observe_preds(self, nodes):
        pred_obs = zeros((self.V, self.D))
        for n in nodes:
            ent = self.ents[n.nodeid, :]
            pred_obs[n.pred, :] += ent * (1 - self.prob(ent, n.pred))
        return pred_obs
    
    def observe_preds_neg(self, nodes):
        pred_neg = zeros((self.V, self.D))
        for n in nodes:
            ent = self.ents[n.nodeid, :]
            
            # Resample the negative preds
            for i, old in enumerate(self.neg_preds[n.nodeid]):
                m = random.randint(self.N)
                new = self.nodes[m].pred
                # Metropolis-Hastings ratio
                ratio = self.freq[new] * self.prob(ent, new) \
                       /self.freq[old] / self.prob(ent, old)
                if ratio > 1:
                    switch = True
                else:
                    switch = random.binomial(1, ratio)
                if switch:
                    self.neg_preds[n.nodeid][i] = new
                else:
                    new = old
                
                # Get gradient
                pred_neg[new, :] += ent * (1 - self.prob(ent, new))
        return pred_neg / self.NEG
    
    def descend(self):
        link_del, pred_del = self.contrast()
        self.link_wei *= self.L2_link
        self.pred_wei *= self.L2_pred
        self.link_wei += self.rate_link * link_del
        self.pred_wei += self.rate_pred * pred_del
        
    
    def train(self, epochs, minibatch=None, print_every=None):
        if minibatch == None:
            minibatch = self.minibatch
        if print_every == None:
            print_every = self.print_every
        G = len(self.graphs)
        indices = arange(G)
        for e in range(epochs):
            # Randomise batches
            # (At the moment, just one batch of particles)
            random.shuffle(indices)
            for i in range(0, G, minibatch):
                batch_graphs = [self.graphs[j] for j in indices[i : i+minibatch]]
                batch = [n for g in batch_graphs for n in g.iter_nodes()]
                # Resample latent variables
                self.resample_metro(batch, self.ents)
                #self.sample_latent_batch(batch)
                self.sample_particle_batch(self.neg_nodes.values())
                neg_link_ratio = len(batch) / self.K
                # Observe latent variables, particles, and negative samples
                link_del = self.observe_links(batch, self.ents)
                link_del -= neg_link_ratio * self.observe_links(self.neg_nodes.values(), self.neg_ents)
                
                """
                # Questionable pred sampling
                pred_del = self.observe_preds(batch)
                pred_del -= self.neg_sample_preds(batch)
                """
                
                """
                #!!!
                # Slow learning at convergence
                pred_del = self.observe_preds_fancy(batch)
                """
                # Correct pred gradients
                pred_del = self.observe_preds(batch)
                pred_del -= self.observe_preds_neg(batch)
                
                # Descend
                self.link_wei += self.rate_link * link_del
                self.pred_wei += self.rate_pred * pred_del
                # Remove negatives
                self.link_wei = self.link_wei.clip(0)
                self.pred_wei = self.pred_wei.clip(0)
                #print(average(absolute(self.pred_wei)), average(absolute(self.rate*pred_del)))
                # Regularise
                self.link_wei *= self.L2_link
                l1_reg(self.link_wei, self.L1_link)
                for n in batch:
                    self.pred_wei[n.pred, :] *= self.L2_pred
                    l1_reg(self.pred_wei[n.pred, :], self.L1_pred)
                self.calc_av_pred()
            # Print regularly
            if e % print_every == 0:
                print(self.link_wei)
                print(self.pred_wei)
                print(amax(absolute(self.link_wei)))
                print(amax(absolute(self.pred_wei)))
                print(self.average_energy())
                with open('../data/out.pkl','wb') as f:
                    pickle.dump(self, f)
                
                rr = [y/x for x,y in self.approx]
                N = len(rr)
                more = len([x for x in rr if x>1])
                doub = len([x for x in rr if x>2])
                half = len([x for x in rr if x<1/2])
                ten = len([x for x in rr if x>10])
                tth = len([x for x in rr if x<1/10])
                print(sum(rr)/N)
                print('more', more/N)
                print('doub', doub/N)
                print('half', half/N)
                print('ten', ten/N)
                print('tth', tth/N)
                
                print('max one:')
                rr = [min(y,1)/min(x,1) for x,y in self.approx]
                N = len(rr)
                more = len([x for x in rr if x>1])
                doub = len([x for x in rr if x>2])
                half = len([x for x in rr if x<1/2])
                ten = len([x for x in rr if x>10])
                tth = len([x for x in rr if x<1/10])
                print(sum(rr)/N)
                print('more', more/N)
                print('doub', doub/N)
                print('half', half/N)
                print('ten', ten/N)
                print('tth', tth/N)
                
                self.approx = []
        
    def train_alternate(self, epochs, minibatch=None, print_every=None, streak=3):
        if minibatch == None:
            minibatch = self.minibatch
        if print_every == None:
            print_every = self.print_every
        G = len(self.graphs)
        indices = arange(G)
        for e in range(epochs):
            # Optimise links
            for _ in range(streak):
                # Randomise batches
                # (At the moment, just one batch of particles)
                random.shuffle(indices)
                for i in range(0, G, minibatch):
                    batch_graphs = [self.graphs[j] for j in indices[i : i+minibatch]]
                    batch = [n for g in batch_graphs for n in g.iter_nodes()]
                    # Resample latent variables
                    self.sample_latent_batch(batch)
                    self.sample_particle_batch(self.neg_nodes.values())
                    neg_link_ratio = len(batch) / self.K
                    # Observe latent variables, particles, and negative samples
                    link_del = self.observe_links(batch, self.ents)
                    link_del -= neg_link_ratio * self.observe_links(self.neg_nodes.values(), self.neg_ents)
                    # Descend
                    self.link_wei += self.rate_link * link_del
                    #print(average(absolute(self.pred_wei)), average(absolute(self.rate*pred_del)))
                    # Regularise
                    self.link_wei *= self.L2_link
                    l1_reg(self.link_wei, self.L1_link)
            # Optimise preds
            for _ in range(streak):
                random.shuffle(indices)
                for i in range(0, G, minibatch):
                    batch_graphs = [self.graphs[j] for j in indices[i : i+minibatch]]
                    batch = [n for g in batch_graphs for n in g.iter_nodes()]
                    # Resample latent variables
                    self.sample_latent_batch(batch)
                    neg_link_ratio = len(batch) / self.K
                    # Observe latent variables, particles, and negative samples
                    pred_del = self.observe_preds(batch)
                    pred_del -= self.neg_sample_preds(batch)
                    # Descend
                    self.pred_wei += self.rate_pred * pred_del
                    #print(average(absolute(self.pred_wei)), average(absolute(self.rate*pred_del)))
                    # Regularise
                    for n in batch:
                        self.pred_wei[n.pred, :] *= self.L2_pred
                        l1_reg(self.pred_wei[n.pred, :], self.L1_pred)
            # Print regularly
            if e % print_every == 0:
                print(self.link_wei)
                print(self.pred_wei)
    
    # Testing functions
    
    def energy(self, graph, entities, pred=True):
        e = 0
        for n in graph.nodes:
            for link in n.outgoing:
                e -= tensordot(tensordot(self.link_wei[link.rargname, :,:],
                                         entities[n.nodeid, :], (0,0)),
                               entities[link.end, :], (0,0))
            if pred:
                e -= tensordot(entities[n.nodeid, :],
                               self.pred_wei[n.pred, :], (0,0))
        return e
    
    def average_energy(self):
        e = 0
        for g in self.graphs.values():
            e += self.energy(g, self.ents)
        return e / len(self.graphs)
    
    def sample_energy(self, graph, samples=5, burnin=5, interval=2, pred=True):
        e = 0
        raw_ents = zeros((len(graph), self.D))
        index = {n.nodeid:i for i,n in enumerate(graph.iter_nodes())}
        ents = WrappedVectors(raw_ents, index)
        for i in range(-burnin, 1+(samples-1)*interval):
            self.resample(graph.iter_nodes(), ents, pred=pred)
            if i >= 0 and i % interval == 0:
                e += self.energy(graph, ents, pred=pred)
        return e/samples
    
    def cosine_of_parameters(self, pred1, pred2):
        return cosine(self.pred_wei[pred1, :],
                      self.pred_wei[pred2, :])
    
    def cosine_samples(self, pred1, pred2, samples=5):
        total = 0
        ents = zeros((2, self.D))
        nodes = [PointerNode(0, pred1), PointerNode(1, pred2)]
        for _ in range(samples):
            self.resample(nodes, ents)
            total += cosine(ents[0,:], ents[1,:])
        return total/samples 
    
    def implies(self, pred1, pred2, samples=5):
        total = 0
        ents = zeros((1, self.D))
        nodes = [PointerNode(0, pred1)]
        for _ in range(samples):
            self.resample(nodes, ents)
            total += self.prob(ents[0,:], pred2)
        return total/samples
    
    def prob(self, ent, pred):
        return expit(dot(ent, self.pred_wei[pred, :]) + self.bias)

class WrappedVectors():
    """
    Access vectors according to different indices
    """
    def __init__(self, matrix, index):
        self.matrix = matrix
        self.index = index
    def __getitem__(self, key):
        return self.matrix[self.index[key[0]], key[1]]
    def __setitem__(self, key, value):
        self.matrix[self.index[key[0]], key[1]] = value

def l1_reg(array, penalty):
    """
    Apply L1 regularisation
    """
    array -= minimum(absolute(array), penalty) * sign(array)