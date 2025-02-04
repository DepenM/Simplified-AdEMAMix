# Simplified-AdEMAMix

This is the official implementation of the Sim-AdEMAMix optimizer. To use, copy the simplified_AdEMAMix.py file to your codebase and use the optimizer in the following fashion (here T represents the total steps of the run):

```
from simplified_AdEMAMix import SimAdEMAMix

optim = SimAdEMAMix(lr = 1e-4, betas=(.99, .95), alpha=0.0, min_beta1=0.9, beta1_warmup=T, weight_decay=0.0)
```

The optimizer by default has the momentum maintained in theory style (not EMA style) with bias correction turned off, which generally seems to help in practice with cosine decay. This also implies that tuning beta1 and learning rate is coupled, in the sense that a higher beta1, should generally scale down the learning rate such that $\eta/(1-\beta_1)$ is constant. Similarly, $\alpha$ really depends on the batch size, and from theory, should scale down linearly with increase in batch size. Our optimal alpha at a batch size of 1m tokens was close to 0 ($\approx 0.05$), while at 32k was close to 100. At higher batch sizes (i.e curvature dominated regime, instead of noise dominated), $\alpha$ should be set close to a small multiple of $1-\beta_1$ (inspired by Nesterov).


