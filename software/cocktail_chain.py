
import math
import numpy as np
import torch

PHYS_DEFAULT = dict(ks=5., c_in=100., c0=None, eps=2e-6, resistant_init=(1e-7, 1e-7, 1e-14),
                    growth_ratio=(1., .9, .9, .8), mutation=True)

def simulate_chain(theta, minutes, k=(16, 16), device='cpu', dt_min=1., dtype=torch.float32, phys=None, return_states=False):
    # Latent period = Erlang(k) with mean L: a chain of k infected stages, each left with probability k*dt/L per step.
    # k = L/dt (probability 1) is a deterministic shift register and equals the fixed-delay solver without superinfection.
    p = {**PHYS_DEFAULT, **(phys or {})}
    kA, kB = (int(k), int(k)) if isinstance(k, (int, np.integer)) else (int(k[0]), int(k[1]))
    theta = theta.to(device=device, dtype=dtype)
    assert theta.ndim == 2 and theta.shape[1] == 15 and kA >= 1 and kB >= 1
    times = np.asarray(minutes, dtype=float)
    assert times[0] == 0 and np.all(np.diff(times) > 0)
    sample_steps = np.rint(times/dt_min).astype(int)
    assert np.allclose(sample_steps*dt_min, times)
    n = len(theta)
    la, lb = theta[:, 4].clamp_min(1e-3), theta[:, 5].clamp_min(1e-3)
    if max(kA*dt_min - float(la.min().detach()), kB*dt_min - float(lb.min().detach())) > 1e-9:
        raise ValueError('k*dt_min must not exceed the shortest latent period')
    fa, fb = (kA*dt_min/la).clamp(max=1.), (kB*dt_min/lb).clamp(max=1.)
    wa = (la < lb).to(dtype) + .5*(la == lb).to(dtype); wb = 1. - wa
    ba, bb = theta[:, 6], theta[:, 7]
    mua, mub = 10.**theta[:, 8], 10.**theta[:, 9]
    if not p['mutation']:
        mua, mub = torch.zeros_like(mua), torch.zeros_like(mub)
    flow, da, db = theta[:, 10], theta[:, 11], theta[:, 12]
    aa, ab = 10.**theta[:, 2], 10.**theta[:, 3]
    pa, pb = 10.**theta[:, 13], 10.**theta[:, 14]
    ks, c_in, eps = float(p['ks']), float(p['c_in']), float(p['eps'])
    growth = theta[:, 1, None]*theta.new_tensor(list(p['growth_ratio']))
    ra, rb, rab = p['resistant_init']
    active = 10.**theta[:, 0, None]*theta.new_tensor([1. - ra - rb - rab, ra, rb, rab])
    nutrient = theta.new_full((n,), c_in if p['c0'] is None else float(p['c0']))
    IA, IB = theta.new_zeros((n, kA)), theta.new_zeros((n, kB))
    dt_h = dt_min/60.; result, states = [], []

    def record():
        infected = IA.sum(1) + IB.sum(1)
        result.append(torch.stack((active.sum(1) + infected, pa, pb), -1).clamp_min(1e-30).log10())
        if return_states:
            states.append(torch.cat((active, infected[:, None]), 1).clamp_min(1e-30).log10())
    record(); cursor = 1
    for step in range(int(sample_steps[-1])):
        nutrient = nutrient + dt_h*flow*(c_in - nutrient)
        monod = nutrient/(nutrient + ks).clamp_min(1e-30)
        new = active*growth*monod[:, None]*dt_h
        active = active + new - active*flow[:, None]*dt_h
        nutrient = (nutrient - eps*new.sum(1)).clamp_min(0.)
        ma, mb = new[:, 0]*mua, new[:, 0]*mub; mab = new[:, 0]*mua*mub
        scale = torch.minimum(torch.ones_like(ma), new[:, 0]/(ma + mb + mab).clamp_min(1e-30))
        ma, mb, mab = ma*scale, mb*scale, mab*scale
        sa, sb = new[:, 1]*mub, new[:, 2]*mua
        active = torch.stack((active[:, 0] - ma - mb - mab, active[:, 1] + ma - sa,
                              active[:, 2] + mb - sb, active[:, 3] + mab + sa + sb), 1).clamp_min(0.)
        survival = (1. - flow*dt_h).clamp_min(0.)
        IA, IB = IA*survival[:, None], IB*survival[:, None]
        oA, oB = IA*fa[:, None], IB*fb[:, None]
        IA = IA - oA + torch.cat((torch.zeros_like(oA[:, :1]), oA[:, :-1]), 1)
        IB = IB - oB + torch.cat((torch.zeros_like(oB[:, :1]), oB[:, :-1]), 1)
        pa = pa + ba*oA[:, -1]; pb = pb + bb*oB[:, -1]
        pa = pa*(1. - (da + flow)*dt_h).clamp_min(0.)
        pb = pb*(1. - (db + flow)*dt_h).clamp_min(0.)
        pool_a = active[:, 0] + active[:, 2]; pool_b = active[:, 0] + active[:, 1]
        raw_a, raw_b = aa*pool_a*pa*dt_min, ab*pool_b*pb*dt_min
        bound_a, bound_b = torch.minimum(raw_a, pa), torch.minimum(raw_b, pb)
        hit_a = torch.minimum(bound_a, pool_a).clamp_min(0.); hit_b = torch.minimum(bound_b, pool_b).clamp_min(0.)
        pa, pb = (pa - bound_a).clamp_min(0.), (pb - bound_b).clamp_min(0.)
        a_s = hit_a*active[:, 0]/pool_a.clamp_min(1e-30); a_r = hit_a*active[:, 2]/pool_a.clamp_min(1e-30)
        b_s = hit_b*active[:, 0]/pool_b.clamp_min(1e-30); b_r = hit_b*active[:, 1]/pool_b.clamp_min(1e-30)
        both = a_s*b_s/active[:, 0].clamp_min(1e-30)
        only_a, only_b = (a_s - both).clamp_min(0.), (b_s - both).clamp_min(0.)
        factor = torch.minimum(torch.ones_like(both), active[:, 0]/(only_a + only_b + both).clamp_min(1e-30))
        only_a, only_b, both = only_a*factor, only_b*factor, both*factor
        a_r, b_r = torch.minimum(a_r, active[:, 2]), torch.minimum(b_r, active[:, 1])
        active = torch.stack((active[:, 0] - only_a - only_b - both, active[:, 1] - b_r,
                              active[:, 2] - a_r, active[:, 3]), 1).clamp_min(0.)
        IA = torch.cat((IA[:, :1] + (only_a + a_r + both*wa)[:, None], IA[:, 1:]), 1)
        IB = torch.cat((IB[:, :1] + (only_b + b_r + both*wb)[:, None], IB[:, 1:]), 1)
        if cursor < len(sample_steps) and step + 1 == sample_steps[cursor]:
            record(); cursor += 1
    out = torch.stack(result, 1)
    if not torch.isfinite(out).all(): raise FloatingPointError('Nonfinite chain-solver output')
    return (out, torch.stack(states, 1)) if return_states else out
