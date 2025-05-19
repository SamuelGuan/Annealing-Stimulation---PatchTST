import torch
from typing import List, Optional, Union, Tuple
from torch import Tensor
from torch.optim.optimizer import (Optimizer, params_t, _use_grad_for_differentiable,
                                   _get_value, _default_to_fused_or_foreach, _dispatch_sqrt)
from torch._utils import is_compiling

__all__ = ['MixOptimizer','adam_asgd']

class MixOptimizer(Optimizer):
    def __init__(self,
                 params: params_t,
                 lr: Union[float, Tensor] = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 lambd=1e-4,
                 alpha=0.75,
                 t0=1e6,
                 amsgrad: bool = False,
                 *,
                 foreach: Optional[bool] = None,
                 maximize: bool = False,
                 capturable: bool = False,
                 differentiable: bool = False,
                 fused: Optional[bool] = None,
                 ):

        # adam 的参数检查
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if isinstance(lr, Tensor) and foreach and not capturable:
            raise ValueError("lr as a Tensor is not supported for capturable=False and foreach=True")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, lambd=lambd,alpha=alpha,initial_lr=lr,
                        t0=t0,betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        maximize=maximize, foreach=foreach, capturable=capturable,
                        differentiable=differentiable, fused=fused)

        super().__init__(params, defaults)


    def __setstate__(self, state):
        '''adam 的状态集合'''
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
            group.setdefault('differentiable', False)
            group.setdefault('fused', None)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))
        eta_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["eta"]
        )
        if not eta_is_tensor:
            for s in state_values:
                s["eta"] = torch.tensor(s["eta"])
        mu_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["mu"]
        )
        if not mu_is_tensor:
            for s in state_values:
                s["mu"] = torch.tensor(float(s["mu"]))

    def _init_group(self,
                    group,
                    params_with_grad,
                    grads,
                    mus,
                    axs,
                    etas,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps
        ):
        '''adam 的初始化方法'''
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    state['step'] = (
                        torch.zeros((), dtype=torch.float, device=p.device)
                        if group['capturable'] or group['fused']
                        else torch.tensor(0.)
                    )
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["eta"] = torch.tensor(group["lr"])
                    state["mu"] = torch.tensor(1.0)
                    state["ax"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                mus.append(state["mu"])
                axs.append(state["ax"])
                etas.append(state["eta"])

                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                if group['differentiable'] and state['step'].requires_grad:
                    raise RuntimeError('`requires_grad` is not supported for `step` in differentiable mode')

                # Foreach without capturable does not support a tensor lr
                if group['foreach'] and torch.is_tensor(group['lr']) and not group['capturable']:
                    raise RuntimeError('lr as a Tensor is not supported for capturable=False and foreach=True')

                state_steps.append(state['step'])

    @_use_grad_for_differentiable
    def step(self, closure=None, epoch_factor=0.8):
        '''
        adam 的 step
        '''
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            mus = []
            axs = []
            etas = []
            state_steps = []
            beta1, beta2 = group['betas']

            self._init_group(
                group=group,
                params_with_grad=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                max_exp_avg_sqs=max_exp_avg_sqs,
                state_steps=state_steps,
                mus=mus, axs=axs, etas=etas)

            adam_asgd(
                params=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                max_exp_avg_sqs=max_exp_avg_sqs,
                state_steps=state_steps,
                amsgrad=group['amsgrad'],
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                axs=axs,
                mus=mus,
                etas=etas,
                lambd=group["lambd"],
                t0=group["t0"],
                alpha=group["alpha"],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group['maximize'],
                foreach=group['foreach'],
                capturable=group['capturable'],
                differentiable=group['differentiable'],
                fused=group['fused'],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                epoch_factor=epoch_factor
            )

        return loss

def adam_asgd(params: List[Tensor],
                grads: List[Tensor],
                exp_avgs: List[Tensor],
                exp_avg_sqs: List[Tensor],
                max_exp_avg_sqs: List[Tensor],
                state_steps: List[Tensor],
                foreach: Optional[bool] = None,
                capturable: bool = False,
                differentiable: bool = False,
                fused: Optional[bool] = None,
                grad_scale: Optional[Tensor] = None,
                found_inf: Optional[Tensor] = None,
                *,
                amsgrad: bool,
                beta1: float,
                beta2: float,
                lr: Union[float, Tensor],
                axs: List[Tensor],
                mus: List[Tensor],
                etas: List[Tensor],
                lambd: float,
                t0: float,
                alpha: float,
                weight_decay: float,
                eps: float,
                maximize: bool,
                epoch_factor: float
              ):
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False

    if foreach is None:
        foreach = False

    if not torch._utils.is_compiling() and not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    func = _single_tensor_adam_asgd

    func(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         amsgrad=amsgrad,
         beta1=beta1,
         beta2=beta2,
         lr=lr,
         axs=axs,
         mus=mus,
         etas=etas,
         lambd=lambd,
         t0=t0,
         alpha=alpha,
         weight_decay=weight_decay,
         eps=eps,
         maximize=maximize,
         capturable=capturable,
         differentiable=differentiable,
         grad_scale=grad_scale,
         found_inf=found_inf,
         epoch_factor=epoch_factor)

MixOptimizer.__doc__ = r"""
    This is an Optimizer combining Adam and ASGD.

    The adam optimizer has the momentum item to accelerate gradient descent.
    However, when it comes to convergence, 
    it will drop into sharp minimum instead of flat minimum with high probability,
    causing the model which is overfit or has worse performance in validation dataset. 

    The SGD optimizer lacks of the momentum item, making it difficult to optimize learnable parameters.
    Yet it was not good at decreasing the loss at the beginning of the training,
    it always overcomes the overfit by his flat loss valley with high probability.

    Reasonably, we can use adam and ASGD's update rule to optimize parameters by some non-linear weight function
    deciding the formula weight in updating.
"""

def _single_tensor_adam_asgd(params: List[Tensor],
                        grads: List[Tensor],
                        exp_avgs: List[Tensor],
                        exp_avg_sqs: List[Tensor],
                        max_exp_avg_sqs: List[Tensor],
                        state_steps: List[Tensor],
                        grad_scale: Optional[Tensor],
                        found_inf: Optional[Tensor],
                        *,
                        amsgrad: bool,
                        beta1: float,
                        beta2: float,
                        lr: Union[float, Tensor],
                        weight_decay: float,
                        eps: float,
                        axs: List[Tensor],
                        mus: List[Tensor],
                        etas: List[Tensor],
                        lambd: float,
                        t0: float,
                        alpha: float,
                        maximize: bool,
                        capturable: bool,
                        differentiable: bool,
                        epoch_factor: float):
    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    epoch_factor1 = epoch_factor
    epoch_factor2 = 1-epoch_factor

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grad
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        mu = mus[i]
        ax = axs[i]
        eta = etas[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch._utils.is_compiling() and capturable:
            assert (
                    (param.is_cuda and step_t.is_cuda) or (param.is_xla and step_t.is_xla)
            ), "If capturable=True, params and state_steps must be CUDA or XLA tensors."

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            param = torch.view_as_real(param)
            ax = torch.view_as_real(ax)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        # update step
        step_t += 1
        step = _get_value(step_t)

        if weight_decay != 0:
            grad_asgd = grad.add(param, alpha=weight_decay)
        else:
            grad_asgd = grad

        eta_value = _get_value(eta)
        # decay term
        param.mul_(1 - lambd * eta_value)

        # update parameter
        param.add_(grad_asgd, alpha=-eta_value*epoch_factor1)

        if capturable or differentiable:
            step = step_t

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

            param.addcdiv_(exp_avg*epoch_factor2, denom)
        else:
            step = _get_value(step_t)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            param.addcdiv_(exp_avg*epoch_factor2, denom, value=-step_size)

        # Lastly, switch back to complex view
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])

        # averaging
        if is_compiling() or mu.item() != 1:
            ax.add_(param.sub(ax).mul(mu))
        else:
            ax.copy_(param)
        new_eta = _to_tensor(lr / ((1 + lambd * lr * step) ** alpha))
        eta.copy_(new_eta)
        new_mu = _to_tensor(1 / max(1, step - t0))
        mu.copy_(new_mu)


def _to_tensor(x):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x)
    return x
