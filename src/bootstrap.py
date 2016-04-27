import pickle
import numpy as np

from main import setup_trainer

trainer = setup_trainer(thresh=5,
                        suffix='bootstrap_strong',
                        dims=300,
                        card=30,
                        init_bias=0,
                        init_card=0,
                        init_range=0,
                        model='independent',
                        rate=0.001,
                        rate_ratio=1,
                        l1=0.01,
                        l1_ratio=1,
                        l1_ent=0,
                        l2=0.01,
                        l2_ratio=1,
                        l2_ent=0.0001,
                        ent_steps=50,
                        pred_steps=5,
                        setup='adagrad',
                        ada_decay=10**-8,
                        neg_samples=1,
                        processes=15,
                        epochs=3,
                        minibatch=20,
                        ent_burnin=50,
                        pred_burnin=5)

# Load sparse vectors
pred_wei = trainer.model.pred_wei
with open('/anfs/bigdisc/gete2/wikiwoods/word2vec/matrix300', 'r') as f:
    for i, line in enumerate(f):
        pred, vecstr = line.strip().split(maxsplit=1)
        assert pred == trainer.model.pred_name[i]
        vec = np.array(vecstr.split())
        pred_wei[i] = vec
# Make vectors longer (av. sum 1.138 over av. 44.9 nonzero entries)
# An average entry is then 0.2, so a predicate is expit(0.2*30 - 3) = 0.95 true
pred_wei *= 8
# Choose biases to match vectors
#(so prob is ~0.5 when half of the 'relevant' units are on)
pred_sums = pred_wei.sum(axis=1)
pred_nonzero = (pred_wei > 0).sum(axis=1)
pred_average = pred_sums / pred_nonzero
pred_maxon = pred_nonzero.clip(0,30)
pred_avfull = pred_average * pred_maxon
trainer.model.pred_bias += pred_avfull / 2

# Remove link weight initialisation
trainer.model.link_wei *= 0
with open('/anfs/bigdisc/gete2/wikiwoods/sem-func/bootstrap_link_log.pkl', 'rb') as f:
    trainer.model.link_wei += pickle.load(f)
trainer.model.ent_bias[:] = 5

# Only train link weights
#trainer.setup.rate_pred = 0

trainer.start()