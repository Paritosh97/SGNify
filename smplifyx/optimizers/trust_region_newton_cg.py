# -*- coding: utf-8 -*-

# @author Vasileios Choutas
# Contact: vassilis.choutas@tuebingen.mpg.de


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from typing import NewType, List, Tuple
import warnings

import torch
import torch.optim as optim
import torch.autograd as autograd

import math

Tensor = NewType('Tensor', torch.Tensor)


class TrustRegionNewtonCG(optim.Optimizer):
    def __init__(self, params: List[Tensor],
                 max_trust_radius: float = 1000,
                 initial_trust_radius: float = 0.05,
                 eta: float = 0.15,
                 gtol: float = 1e-05,
                 **kwargs) -> None:
        defaults = dict()
        super(TrustRegionNewtonCG, self).__init__(params, defaults)

        self.steps = 0
        self.max_trust_radius = max_trust_radius
        self.initial_trust_radius = initial_trust_radius
        self.eta = eta
        self.gtol = gtol
        self._params = self.param_groups[0]['params']

    @torch.enable_grad()
    def _compute_hessian_vector_product(self, gradient: Tensor, p: Tensor) -> Tensor:
        hess_vp = autograd.grad(
            torch.sum(gradient * p, dim=-1), self._params,
            only_inputs=True, retain_graph=True, allow_unused=True)
        return torch.cat(
            [torch.flatten(vp) for vp in hess_vp if vp is not None], dim=-1)

    def _gather_flat_grad(self) -> Tensor:
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        output = torch.cat(views, 0)
        return output

    @torch.no_grad()
    def _improvement_ratio(self, p, start_loss, gradient, closure):
        hess_vp = self._compute_hessian_vector_product(gradient, p)

        with torch.no_grad():
            start_idx = 0
            for param in self._params:
                num_els = param.numel()
                curr_upd = p[start_idx:start_idx + num_els]
                param.data.add_(curr_upd.view_as(param))
                start_idx += num_els

        new_loss = closure(backward=False)
        numerator = start_loss - new_loss
        new_quad_val = self._quad_model(p, start_loss, gradient, hess_vp)
        denominator = start_loss - new_quad_val
        ratio = numerator / (denominator + 1e-20)
        return ratio

    @torch.no_grad()
    def _quad_model(self, p: Tensor, loss: float, gradient: Tensor, hess_vp: Tensor) -> float:
        return (loss + torch.flatten(gradient * p).sum(dim=-1) +
                0.5 * torch.flatten(hess_vp * p).sum(dim=-1))

    @torch.no_grad()
    def calc_boundaries(self, iterate: Tensor, direction: Tensor, trust_radius: float) -> Tuple[Tensor, Tensor]:
        a = torch.sum(direction ** 2, dim=-1)
        b = 2 * torch.sum(direction * iterate, dim=-1)
        c = torch.sum(iterate ** 2, dim=-1) - trust_radius ** 2
        sqrt_discriminant = torch.sqrt(b * b - 4 * a * c)
        ta = (-b + sqrt_discriminant) / (2 * a)
        tb = (-b - sqrt_discriminant) / (2 * a)
        if ta.item() < tb.item():
            return [ta, tb]
        else:
            return [tb, ta]

    @torch.no_grad()
    def _solve_trust_reg_subproblem(self, loss: float, flat_grad: Tensor, trust_radius: float) -> Tuple[Tensor, bool]:
        iterate = torch.zeros_like(flat_grad, requires_grad=False)
        residual = flat_grad.detach()
        direction = -residual

        if torch.isnan(residual).any():
            raise RuntimeError("NaN detected in the residual vector before starting iterations")

        jac_mag = torch.norm(flat_grad).item()
        tolerance = min(0.5, math.sqrt(jac_mag)) * jac_mag

        if jac_mag <= tolerance:
            return iterate, False

        while True:
            try:
                hessian_vec_prod = self._compute_hessian_vector_product(flat_grad, direction)

                if torch.isnan(hessian_vec_prod).any():
                    raise RuntimeError("NaN detected in the Hessian-vector product")

                hevp_dot_prod = torch.sum(hessian_vec_prod * direction)

                if hevp_dot_prod.item() <= 0:
                    ta, tb = self.calc_boundaries(iterate, direction, trust_radius)
                    pa = iterate + ta * direction
                    pb = iterate + tb * direction

                    bound1_val = self._quad_model(pa, loss, flat_grad, hessian_vec_prod)
                    bound2_val = self._quad_model(pb, loss, flat_grad, hessian_vec_prod)

                    if bound1_val.item() < bound2_val.item():
                        return pa, True
                    else:
                        return pb, True

                residual_sq_norm = torch.sum(residual * residual, dim=-1)
                cg_step_size = residual_sq_norm / hevp_dot_prod
                next_iterate = iterate + cg_step_size * direction
                iterate_norm = torch.norm(next_iterate, dim=-1)

                if iterate_norm.item() >= trust_radius:
                    ta, tb = self.calc_boundaries(iterate, direction, trust_radius)
                    p_boundary = iterate + tb * direction
                    return p_boundary, True

                next_residual = residual + cg_step_size * hessian_vec_prod

                if torch.isnan(next_residual).any():
                    raise RuntimeError("NaN detected in the next residual")

                if torch.norm(next_residual, dim=-1).item() < tolerance:
                    return next_iterate, False

                beta = torch.sum(next_residual ** 2, dim=-1) / residual_sq_norm
                direction = (-next_residual + beta * direction).squeeze()

                if torch.isnan(direction).any():
                    raise RuntimeError("NaN detected in the direction vector")

                iterate = next_iterate
                residual = next_residual

            except RuntimeError as e:
                print(f"RuntimeError in _solve_trust_reg_subproblem: {e}")
                print(f"iterate: {iterate}")
                print(f"residual: {residual}")
                print(f"direction: {direction}")
                print(f"trust_radius: {trust_radius}")
                raise

    def step(self, closure) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            starting_loss = closure(backward=True)

        flat_grad = self._gather_flat_grad()

        state = self.state
        if len(state) == 0:
            state['trust_radius'] = torch.full([1],
                                               self.initial_trust_radius,
                                               dtype=flat_grad.dtype,
                                               device=flat_grad.device)
        trust_radius = state['trust_radius']

        param_step, hit_boundary = self._solve_trust_reg_subproblem(
            starting_loss, flat_grad, trust_radius)
        self.param_step = param_step

        if torch.norm(param_step).item() <= self.gtol:
            return starting_loss

        improvement_ratio = self._improvement_ratio(
            param_step, starting_loss, flat_grad, closure)

        if improvement_ratio.item() < 0.25:
            trust_radius.mul_(0.25)
        else:
            if improvement_ratio.item() > 0.75 and hit_boundary:
                trust_radius.mul_(2).clamp_(0.0, self.max_trust_radius)

        if improvement_ratio.item() <= self.eta:
            start_idx = 0
            for param in self._params:
                num_els = param.numel()
                curr_upd = param_step[start_idx:start_idx + num_els]
                param.data.add_(-curr_upd.view_as(param))
                start_idx += num_els

        self.steps += 1
        return starting_loss