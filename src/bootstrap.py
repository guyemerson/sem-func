import pickle
import numpy as np

from main import setup_trainer
# Load sparse vectors
from bootstrap_meanfield import pred_wei, D, C

trainer = setup_trainer(thresh=5,
                        suffix='bootstrap_bigratio',
                        dims=D,
                        card=C,
                        init_bias=0,
                        init_card=0,
                        init_range=0,
                        model='independent',
                        rate=0.01,
                        rate_ratio=1,
                        l1=0.00001,
                        l1_ratio=0.01,
                        l1_ent=0,
                        l2=0.001,
                        l2_ratio=0.01,
                        l2_ent=0.0001,
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
trainer.model.pred_wei[:] = pred_wei
pred_wei = trainer.model.pred_wei
pred_bias = trainer.model.pred_bias

# Choose biases to match vectors
#(so prob is high when most relevant units are on, and low when random units are on)
thresh = 5
high = np.partition(pred_wei, -C, axis=1)[:,-C:].sum(axis=1)
aver = pred_wei.mean(axis=1) * C * 2

clip_mask = (high - aver > 2 * thresh)
#pred_bias[:] = clip_mask * pred_bias.clip(aver+5, high-5) + (1 - clip_mask) * (high-aver)/2  # Seems to be slower than:
pred_bias[clip_mask].clip(aver[clip_mask]+thresh, high[clip_mask]-thresh, out=pred_bias[clip_mask])
diff_mask = 1 - clip_mask
pred_bias[diff_mask] = (high[diff_mask] - aver[diff_mask])/2

# Remove link weight initialisation
with open('/anfs/bigdisc/gete2/wikiwoods/sem-func/bootstrap_link_log400.pkl', 'rb') as f:
    trainer.model.link_wei[:] = pickle.load(f)
trainer.model.ent_bias[:] = 5

trainer.start()