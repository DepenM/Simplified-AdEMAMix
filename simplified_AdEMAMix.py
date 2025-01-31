"""
Adapted from: https://github.com/apple/ml-ademamix
"""
import math
import torch
from torch.optim import Optimizer


def linear_hl_warmup_scheduler(step, beta_end, beta_start=0, warmup=1):

    def f(beta, eps=1e-8):
        return math.log(0.5)/math.log(beta+eps)-1

    def f_inv(t):
        return math.pow(0.5, 1/(t+1))

    if step < warmup:
        a = step / float(warmup)
        return f_inv((1.0-a) * f(beta_start) + a * f(beta_end))
    return beta_end


class SimAdEMAMix(Optimizer):
    r"""Implements the Simplified AdEMAMix algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999, 0.9999)) 
            corresponding to beta_1, beta_2, beta_3 in AdEMAMix
        alpha (float): AdEMAMix alpha coeficient mixing the slow and fast EMAs (default: 2)
        beta3_warmup (int, optional): number of warmup steps used to increase beta3 (default: None)
        alpha_warmup: (int, optional): number of warmup steps used to increase alpha (default: None)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay as in AdamW (default: 0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.99, 0.999), alpha=2.0, 
                 beta1_warmup=None, eps=1e-8, weight_decay=0.0, min_beta1=0.0,
                 bias_correction1=False, bias_correction2=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        defaults = dict(lr=lr, betas=betas, eps=eps, alpha=alpha, beta1_warmup=beta1_warmup,
                        weight_decay=weight_decay, min_beta1=min_beta1)
        super(SimAdEMAMix, self).__init__(params, defaults)
        self._bias_correction1 = bias_correction1
        self._bias_correction2 = bias_correction2

    def __setstate__(self, state):
        super(SimAdEMAMix, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            
            lr = group["lr"]
            lmbda = group["weight_decay"]
            eps = group["eps"]
            beta1_final, beta2 = group["betas"]
            beta1_warmup = group["beta1_warmup"]
            alpha = group["alpha"]
        
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdEMAMix does not support sparse gradients.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['num_sum'] = 0.0
                    state['den_sum'] = 0.0

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                
                if beta1_warmup is not None:
                    beta1 = linear_hl_warmup_scheduler(state["step"], beta_end=beta1_final, beta_start=group['min_beta1'], warmup=beta1_warmup)
                else:
                    beta1 = beta1_final
                        
                exp_avg.mul_(beta1).add_(grad, alpha=1.0)
                state['num_sum'] = beta1 * state['num_sum'] + 1.0
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                state['den_sum'] = beta2 * state['den_sum'] + (1.0 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(eps*math.sqrt(state['den_sum']))

                update = (alpha * grad + exp_avg) / denom
                if self._bias_correction1:
                    update = update/state['num_sum']
                if self._bias_correction2:
                    update = update*math.sqrt(state['den_sum'])

                # decay
                update.add_(p, alpha=lmbda)

                p.add_(-lr * update)

        return loss


if __name__ == "__main__": # small dummy test

    x = torch.randn((10,7))
    model = torch.nn.Linear(7, 1, bias=False)
    opt = SimAdEMAMix(params=model.parameters(), lr=1e-2, betas=(0.9, 0.999), alpha=2.0, beta1_warmup=45, weight_decay=0.1)
    print(model.weight)
    for itr in range(50):
        y = model(x).mean()
        opt.zero_grad()
        y.backward()
        opt.step()

    print(model.weight)