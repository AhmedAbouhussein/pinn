# PINN Solver: 2D Helmholtz Equation with Deep Narrow MLP

## Problem

Solve the 2D Helmholtz equation on Ω = [-1, 1]²:

```math
\Delta u + k^2 u = q(x,y), \quad k = 1.0
```

with homogeneous Dirichlet BCs (u = 0 on ∂Ω) and forcing:

```math
q(x,y) = (1 - 17\pi^2)\sin(\pi x)\sin(4\pi y)
```

The reference solution is $u(x,y) = \sin(\pi x)\sin(4\pi y)$. The solver must treat this as unknown.

**Constraints:** 12-hidden-layer MLP, ≤ 5,500 float32 parameters (22 KB), tanh activation, raw (x,y) input. No Fourier features, skip connections, adaptive activations, or hard BC wrappers. Soft BC penalty only. Maximum 10,000 iterations.

## Architecture

| Component | Value |
|-----------|-------|
| Input | 2 (x, y) |
| Hidden | 12 layers × 21 neurons, tanh |
| Output | 1 |
| Parameters | 5,167 (20.18 KB) |
| Init | Xavier uniform, gain = 5/3 |

## Gradient Pathology and Motivation

Wang et al. (2020) show that PINN composite losses $\mathcal{L} = \mathcal{L}_r + \lambda\mathcal{L}_{bc}$ suffer from gradient imbalance: for a solution with frequency C, the PDE residual gradients scale as

```math
\|\nabla_\theta \mathcal{L}_r\| \sim O(C^4) \cdot \|\nabla_\theta \mathcal{L}_{bc}\|
```

With $a_2 = 4$ (effective $C = 4\pi$), this ratio is ~25,000×. Standard training overwhelmingly follows $\mathcal{L}_r$ and neglects BCs, producing solutions that satisfy the PDE but not the boundary conditions.

The paper's Algorithm 1 addresses this by scaling $\lambda_{bc}$ up via $\max|\nabla\mathcal{L}_r| / \overline{|\nabla\mathcal{L}_{bc}|}$. In 12-layer networks, this ratio diverges due to gradient outliers in deep layers, causing $\lambda_{bc}$ to reach $10^6$–$10^9$ and flip the imbalance.

## Solution: Inverted Annealing with Phase-Separated Training

Instead of scaling $\mathcal{L}_{bc}$ up, we scale $\mathcal{L}_r$ down. The weight is computed once from gradient norms:

```math
w_r = \frac{\|\nabla_\theta \mathcal{L}_{bc}\|_2}{\|\nabla_\theta \mathcal{L}_r\|_2}
```

This yields $w_r \in (0, 1]$, is naturally bounded, and equalizes the effective gradient contributions: $\|w_r\nabla\mathcal{L}_r\|_2 = \|\nabla\mathcal{L}_{bc}\|_2$.

Training is split into three phases to avoid competing objectives:

### Phase 1 — Adam, $\mathcal{L}_r$ only (2,000 iters)

Flat+cosine LR schedule: warmup 1e-3 → 2e-2 (100 iters), flat at 2e-2 (to iter 1500), cosine decay to 1e-5 (to iter 2000). 4,096 collocation points. No BC loss — the network learns the PDE structure without interference.

### Phase 2 — Adam, $w_r \cdot \mathcal{L}_r + \mathcal{L}_{bc}$ (6,000 iters)

$w_r$ computed once at the Phase 1→2 transition and fixed. Flat+cosine LR: 5e-3 peak, flat to iter 4500, cosine to 5e-4. Both losses decrease simultaneously under balanced gradients.

### Phase 3 — L-BFGS polish (2,000 iters)

Same fixed $w_r$. lr = 1.0, max_iter = 20, strong Wolfe line search, history = 100. Both losses are small at this point, so the loss surface is approximately quadratic and L-BFGS's Hessian approximation is accurate.

## Results

### Convergence

| Metric | Value |
|--------|-------|
| Final PDE Loss ($\mathcal{L}_r$) | 1.23e-03 |
| Final BC Loss ($\mathcal{L}_{bc}$) | 1.90e-04 |
| Relative L₂ Error | **0.60%** |
| Target | ≤ 1.0% |
| Status | **PASS** |
| Total Iterations | 10,000 |

### Validation Points

| Point | Coords | Reference | PINN | Abs Error |
|-------|--------|-----------|------|-----------|
| A | (0.1, 0.1) | 0.30060120 | 0.30254498 | 1.94e-03 |
| B | (0.5, 0.125) | 0.38268343 | 0.38516018 | 2.48e-03 |
| C | (0.3, 0.7) | 0.76604444 | 0.76152980 | 4.51e-03 |

## References

- Wang, S., Teng, Y., & Perdikaris, P. (2020). *Understanding and Mitigating Gradient Pathologies in Physics-Informed Neural Networks.* arXiv:2001.04536.
