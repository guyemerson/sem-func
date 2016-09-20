import pickle, os.path
import numpy as np

from main import setup_trainer
# Load sparse vectors
from bootstrap_meanfield import pred_wei, D, C
from __config__.filepath import INIT_DIR

trainer = setup_trainer(thresh=5,
                        suffix='SCRAP',
                        dims=D,
                        card=C,
                        init_bias=0,
                        init_card=100,
                        init_range=10,
                        model='independent',
                        rate=0.1,
                        rate_ratio=1,
                        l1=0.00001,
                        l1_ratio=0.01,
                        l1_ent=0,
                        l2=0.00001,
                        l2_ratio=0.01,
                        l2_ent=10**-8,
                        ent_steps=50,
                        pred_steps=5,
                        setup='adagrad',
                        ada_decay=10**-4,
                        neg_samples=1,
                        processes=10,
                        epochs=3,
                        minibatch=20,
                        ent_burnin=200,
                        pred_burnin=5)

# Copy vectors
trainer.model.pred_wei[:] = pred_wei * 10
pred_wei = trainer.model.pred_wei
pred_bias = trainer.model.pred_bias

# Choose biases to match vectors
#(so prob is high when most relevant units are on, and low when random units are on)
thresh = 30
high = np.partition(pred_wei, -C, axis=1)[:,-C:].sum(axis=1)
aver = pred_wei.mean(axis=1) * C * 2

clip_mask = (high - aver > 2 * thresh)
#pred_bias[:] = clip_mask * pred_bias.clip(aver+5, high-5) + (1 - clip_mask) * (high-aver)/2  # Seems to be slower than:
pred_bias[clip_mask].clip(aver[clip_mask]+thresh, high[clip_mask]-thresh, out=pred_bias[clip_mask])
diff_mask = 1 - clip_mask
pred_bias[diff_mask] = (high[diff_mask] - aver[diff_mask])/2


# Remove link weight initialisation
with open(os.path.join(INIT_DIR, 'bootstrap_link_log400.pkl'), 'rb') as f:
    trainer.model.link_wei[:] = pickle.load(f) * 10
trainer.model.ent_bias[:] = 60



print("Profiling with one process on one file...")
import cProfile, os
file = os.listdir(trainer.data_dir)[0]
cProfile.runctx('trainer.train_on_file(file)', globals(), locals())