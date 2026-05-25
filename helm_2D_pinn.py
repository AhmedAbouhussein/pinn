"""
2D Helmholtz PINN
  Δu + k²u = q(x,y)  on Ω = [-1,1]²
  u = 0               on ∂Ω
  u(x,y) = sin(πx)sin(4πy),  k=1

Three-phase strategy:
  Phase 1 (Adam, 2000 iters) — L_r only:
    - Warmup 1e-3 → 2e-2 (100), flat 2e-2 (1400, to iter 1500),
      cosine→1e-5 (500, to iter 2000)
    - 4096 collocation points
  Phase 2 (Adam, 6000 iters) — w_r * L_r + L_bc:
    - w_r computed once at transition
    - Flat+cosine: warmup 1e-3→5e-3 (100), flat 5e-3 (4400, to iter 4500),
      cosine→5e-4 (1500, to iter 6000)
  Phase 3 (L-BFGS, 2000 iters) — polish:
    - Same fixed w_r, lr=1.0, max_iter=20
"""
#%%
import torch
import torch.nn as nn
import numpy as np
import math
import time
#%%
torch.manual_seed(42)
np.random.seed(42)

k_sq = 1.0
a1, a2 = 1, 4

#%%
def u_exact(x, y):
    return np.sin(a1 * np.pi * x) * np.sin(a2 * np.pi * y)

class MLP(nn.Module):
    def __init__(self, width=21, depth=12):
        super().__init__()
        layers = [nn.Linear(2, width)]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
        layers.append(nn.Linear(width, 1))
        self.layers = nn.ModuleList(layers)
        for m in self.layers:
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(m.bias)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)

def sample_interior(n):
    x = torch.rand(n, 1) * 2 - 1
    y = torch.rand(n, 1) * 2 - 1
    return torch.cat([x, y], dim=1)

def sample_boundary(n_per_side):
    pts = []
    t = torch.linspace(-1, 1, n_per_side)
    pts.append(torch.stack([t, -torch.ones_like(t)], dim=1))
    pts.append(torch.stack([t,  torch.ones_like(t)], dim=1))
    pts.append(torch.stack([-torch.ones_like(t), t], dim=1))
    pts.append(torch.stack([ torch.ones_like(t), t], dim=1))
    return torch.cat(pts, dim=0)

def compute_losses(model, xy_int, xy_bc):
    xy = xy_int.clone().requires_grad_(True)
    u = model(xy)
    grads = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    u_x, u_y = grads[:, 0:1], grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xy, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xy, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    x_v, y_v = xy[:, 0:1], xy[:, 1:2]
    q = (k_sq - (a1*np.pi)**2 - (a2*np.pi)**2) * \
        torch.sin(a1*np.pi*x_v) * torch.sin(a2*np.pi*y_v)
    residual = u_xx + u_yy + k_sq * u - q
    loss_r = torch.mean(residual**2)
    u_bc = model(xy_bc)
    loss_bc = torch.mean(u_bc**2)
    return loss_r, loss_bc

def compute_wr(model, loss_r, loss_bc):
    """w_r = ||∇θL_bc|| / ||∇θL_r||"""
    grads_r = torch.autograd.grad(loss_r, model.parameters(),
                                   retain_graph=True, allow_unused=True)
    grads_bc = torch.autograd.grad(loss_bc, model.parameters(),
                                    retain_graph=True, allow_unused=True)
    all_r = torch.cat([g.flatten() for g in grads_r if g is not None])
    all_bc = torch.cat([g.flatten() for g in grads_bc if g is not None])
    norm_r = all_r.norm().item()
    norm_bc = all_bc.norm().item()
    if norm_r < 1e-30:
        return 1.0
    return norm_bc / norm_r

def flat_cosine_lr(it, warmup_end, flat_end, total, lr_max, lr_init, lr_min):
    """Warmup → flat → cosine decay."""
    if it <= warmup_end:
        return lr_init + (lr_max - lr_init) * (it / warmup_end)
    elif it <= flat_end:
        return lr_max
    else:
        progress = (it - flat_end) / (total - flat_end)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

def eval_error(model):
    np.random.seed(123)
    x = np.random.uniform(-1, 1, 100)
    y = np.random.uniform(-1, 1, 100)
    ref = u_exact(x, y)
    xy = torch.tensor(np.column_stack([x, y]), dtype=torch.float32)
    with torch.no_grad():
        pred = model(xy).numpy().flatten()
    return np.linalg.norm(pred - ref) / np.linalg.norm(ref)


def train():
    model = MLP(width=21, depth=12)
    total_params = sum(p.numel() for p in model.parameters())
    total_kb = total_params * 4 / 1024
    print(f"Architecture: 12 hidden layers x 21 neurons, tanh")
    print(f"Total parameters: {total_params}")
    print(f"Memory: {total_kb:.2f} KB (limit: 22 KB)")
    assert total_params <= 5500 and total_kb < 22
    print()

    xy_bc = sample_boundary(100)  # 400 boundary points

    # ═══ Phase 1: Adam — L_r only (2000 iters) ═══
    n_phase1 = 2000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Phase 1: Adam ({n_phase1} iters) — L_r only")
    print(f"  Schedule: warmup 1e-3→2e-2 (100), flat 2e-2 to 1500, cosine→1e-5 to 2000")
    print("-" * 65)

    for it in range(1, n_phase1 + 1):
        lr_now = flat_cosine_lr(it, warmup_end=100, flat_end=1500, total=2000,
                                lr_max=2e-2, lr_init=1e-3, lr_min=1e-5)
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now

        xy_int = sample_interior(4096)
        loss_r, loss_bc = compute_losses(model, xy_int, xy_bc)

        optimizer.zero_grad()
        loss_r.backward()
        optimizer.step()

        if it % 500 == 0 or it == 1:
            err = eval_error(model)
            print(f"  Iter {it:5d} | L_r: {loss_r.item():.4e} | "
                  f"L_bc: {loss_bc.item():.4e} | "
                  f"lr: {lr_now:.2e} | err: {err:.4e}")

    # Compute w_r at transition
    xy_int_probe = sample_interior(4096)
    loss_r_probe, loss_bc_probe = compute_losses(model, xy_int_probe, xy_bc)
    w_r = compute_wr(model, loss_r_probe, loss_bc_probe)
    print(f"\n  L_r at end of Phase 1: {loss_r_probe.item():.4e}")
    print(f"  L_bc at end of Phase 1: {loss_bc_probe.item():.4e}")
    print(f"  Norm-based w_r (FIXED): {w_r:.6f}")
    print()

    # ═══ Phase 2: Adam — w_r * L_r + L_bc (6000 iters) ═══
    n_phase2 = 6000

    optimizer2 = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Phase 2: Adam ({n_phase2} iters) — w_r*L_r + L_bc, w_r={w_r:.6f} (fixed)")
    print(f"  Schedule: warmup 1e-3→5e-3 (100), flat 5e-3 to 4500, cosine→5e-4 to 6000")
    print("-" * 65)

    for it in range(1, n_phase2 + 1):
        lr_now = flat_cosine_lr(it, warmup_end=100, flat_end=4500, total=6000,
                                lr_max=5e-3, lr_init=1e-3, lr_min=5e-4)
        for pg in optimizer2.param_groups:
            pg['lr'] = lr_now

        xy_int = sample_interior(4096)
        loss_r, loss_bc = compute_losses(model, xy_int, xy_bc)
        total = w_r * loss_r + loss_bc

        optimizer2.zero_grad()
        total.backward()
        optimizer2.step()

        if it % 1000 == 0 or it == 1:
            err = eval_error(model)
            print(f"  Iter {it:5d} | L_r: {loss_r.item():.4e} | "
                  f"L_bc: {loss_bc.item():.4e} | "
                  f"lr: {lr_now:.2e} | err: {err:.4e}")

    print()

    # ═══ Phase 3: L-BFGS polish (2000 iters) ═══
    n_phase3 = 2000
    xy_int_fixed = sample_interior(4096)

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=20,
        history_size=100,
        line_search_fn="strong_wolfe",
        lr=1.0,
        tolerance_grad=1e-16,
        tolerance_change=1e-16,
    )

    print(f"Phase 3: L-BFGS polish ({n_phase3} iters, w_r={w_r:.6f}, lr=1.0, max_iter=20)")
    print("-" * 65)

    for it in range(1, n_phase3 + 1):
        def closure():
            optimizer_lbfgs.zero_grad()
            lr, lbc = compute_losses(model, xy_int_fixed, xy_bc)
            total = w_r * lr + lbc
            total.backward()
            return total

        optimizer_lbfgs.step(closure)

        if it % 500 == 0 or it == 1:
            lr_v, lbc_v = compute_losses(model, xy_int_fixed, xy_bc)
            err = eval_error(model)
            print(f"  L-BFGS {it:5d} | L_r: {lr_v.item():.4e} | "
                  f"L_bc: {lbc_v.item():.4e} | err: {err:.4e}")

    # ═══ Final Report ═══
    print()
    print("=" * 65)
    lr_f, lbc_f = compute_losses(model, xy_int_fixed, xy_bc)
    print(f"CONVERGENCE LOG:")
    print(f"  Final PDE Loss  (L_r):  {lr_f.item():.6e}")
    print(f"  Final BC  Loss  (L_bc): {lbc_f.item():.6e}")

    rel_l2 = eval_error(model)
    status = "PASS" if rel_l2 <= 0.01 else "FAIL"
    print(f"\nRelative L2 error (100 pts): {rel_l2:.6e}  [{status}, target <= 1.0%]")
    print()

    points = {"A": (0.1, 0.1), "B": (0.5, 0.125), "C": (0.3, 0.7)}
    print(f"{'Point':<8} {'Coords':<16} {'Reference':>12} {'PINN':>12} {'Abs Error':>12}")
    print("-" * 65)
    for name, (px, py) in points.items():
        ref = u_exact(px, py)
        with torch.no_grad():
            pred = model(torch.tensor([[px, py]], dtype=torch.float32)).item()
        print(f"{name:<8} ({px}, {py}){'':<6} {ref:>12.8f} {pred:>12.8f} {abs(pred-ref):>12.2e}")

    print()
    print(f"ARCHITECTURE DETAIL:")
    print(f"  Layers: 2 -> [21]x12 -> 1, tanh")
    print(f"  Parameters: {total_params} ({total_kb:.2f} KB)")
    print(f"  Iterations: {n_phase1} + {n_phase2} + {n_phase3} = {n_phase1 + n_phase2 + n_phase3}")

    return model, rel_l2

#%%
if __name__ == "__main__":
    t0 = time.time()
    model, error = train()
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
# %%
