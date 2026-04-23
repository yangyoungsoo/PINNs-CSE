# PINN Playground

A small workspace containing

1. a self-contained PINN experiment for the 2D Poisson equation (notebook + model module), and
2. helper notebooks for aggregating benchmark accuracy results.

## Contents

| File | Purpose |
|---|---|
| `model.py` | PINN ansatz (`PINN` class) — a 2-50-50-50-50-1 tanh MLP with Xavier init. |
| `poisson_pinn.ipynb` | End-to-end experiment: problem statement, sampling, loss, training, evaluation and plots. |
| `accuracy.csv` | Benchmark results (one row per run) used as input to the stats notebook. |
| `stat.ipynb` | Aggregates `accuracy.csv` — per-problem / per-model mean `relL2`, and mean elapsed time & error on `convection1d`. |

## The Poisson 2D experiment

Solves the Dirichlet boundary-value problem

$$
-\Delta u(x, y) = \sin(\pi x)\sin(\pi y), \qquad (x, y) \in (0, 1)^2,
\qquad u\big|_{\partial\Omega} = 0,
$$

whose analytical solution is

$$
u^{\star}(x, y) = \frac{1}{2\pi^{2}}\sin(\pi x)\sin(\pi y).
$$

The PINN minimizes

$$
\mathcal{L} = \mathcal{L}_{\text{PDE}} + \lambda_b\,\mathcal{L}_{\text{BC}},
$$

with $\mathcal{L}_{\text{PDE}}$ the MSE of the residual $u_{xx} + u_{yy} + f$ on $10{,}000$ interior collocation points, $\mathcal{L}_{\text{BC}}$ the MSE of $u$ against $0$ on $400$ boundary points (100 per edge), and $\lambda_b = 1$. Training uses full-batch Adam at $lr = 10^{-3}$ for $10{,}000$ iterations; progress is streamed with `tqdm`. After training, the notebook reports the relative $L^2$ error and the max absolute error against the closed-form reference, then shows a three-panel comparison (prediction / reference / absolute error) plus the loss curves.

## Setup

```bash
pip install torch numpy matplotlib tqdm pandas
```

`torch` runs on CPU or GPU — the notebook selects CUDA automatically if available. Pandas is only needed for `stat.ipynb`.

## Run order

1. Open `poisson_pinn.ipynb` and run the cells top-to-bottom. On a laptop CPU training takes roughly a minute; on a GPU it is a few seconds.
2. (Optional) Open `stat.ipynb` to aggregate `accuracy.csv`.

## Extending the experiment

- **Different source / BC** — edit the `f = ...` line in the PDE residual cell and (if non-homogeneous) adjust the boundary target in `total_loss`.
- **Different architecture** — pass a custom `layers` list to `PINN(layers=[...])` in the model-instantiation cell.
- **Sharper convergence** — append an L-BFGS phase after Adam, or raise `LAMBDA_B` if the boundary is under-fit.
