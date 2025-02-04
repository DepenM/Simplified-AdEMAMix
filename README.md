# Simplified-AdEMAMix

This is the official implementation of the Sim-AdEMAMix optimizer. To use, copy the simplified_AdEMAMix.py file to your codebase and use the optimizer in the following fashion (here T represents the total steps of the run):

```
from simplified_AdEMAMix import SimAdEMAMix

optim = SimAdEMAMix(lr = 1e-4, betas=(.99, .95), alpha=0.0, min_beta1=0.9, beta1_warmup=T, weight_decay=0.0)
```

The optimizer by default has the momentum maintained in theory style (not EMA style) with bias correction turned off, which generally seems to help in practice with cosine decay. Optimal value of $\alpha$ really depends on the batch size, and from theory, should scale down linearly with increase in batch size. Our optimal alpha at a batch size of 1m tokens was close to 0 ($\aprox 0.05$), while at 32k was close to 100. At higher batch sizes (i.e curvature dominated regime, instead of noise dominated), $\alpha$ should be set close to a small multiple of $1-\beta_1$ (inspired by Nesterov).

For tuning $\eta, \beta_1, \beta_2$ and min_beta, if we have an optimal Adam run with hyperparameters $\eta^{adam}, \beta_1^{adam}$ and $\beta_2^{adam}$, we recommend that for AdEMAMix, the optimal hyperparameters should be around min_beta = $\beta_1^{adam}$, $\beta_1$ higher than min_beta (for min_beta=0.9, maybe try 0.95, 0.99, 0.999), $\beta_2 = \beta_2^{adam}$ and $\eta = \eta^{adam} \sqrt{(1-min_beta)*(1-\beta_1)}$ (thus optimal $\eta$ is coupled with value of $\beta_1$). One more thing to note is that $\beta_1$ should generally decrease with increasing batch size.


