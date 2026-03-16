# PINNs for Fluid Flow & Hemodynamics 🩸

**Free PyTorch notebooks for Physics-Informed Neural Networks applied to 2D incompressible flow and biomedical hemodynamics.**

No mesh. No finite differences. Just neural networks trained to satisfy Navier-Stokes.

[![Open 01 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/menonpg/pinns-hemodynamics/blob/main/01_lid_driven_cavity.ipynb)
[![Open 02 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/menonpg/pinns-hemodynamics/blob/main/02_poiseuille_blood_vessel.ipynb)
[![Open 03 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/menonpg/pinns-hemodynamics/blob/main/03_carotid_bifurcation.ipynb)

---

## Notebooks

| # | Notebook | Topic | Benchmark |
|---|---|---|---|
| 01 | `01_lid_driven_cavity` | Lid-driven cavity Re=100 | Ghia et al. (1982) |
| 02 | `02_poiseuille_blood_vessel` | Straight vessel, Poiseuille flow + WSS | Analytical solution |
| 03 | `03_carotid_bifurcation` | Carotid Y-bifurcation, WSS atherogenic zones | Clinical reference ranges |

---

## What are PINNs?

Standard neural networks learn from **data**. Physics-Informed Neural Networks learn from **equations** — the governing PDEs are baked directly into the training loss via automatic differentiation.

For incompressible Navier-Stokes:

```
Loss = PDE residuals + boundary condition violations

PDE residuals:
  Continuity:   ∂u/∂x + ∂v/∂y = 0
  x-Momentum:   u∂u/∂x + v∂u/∂y = -∂p/∂x + (1/Re)∇²u  
  y-Momentum:   u∂v/∂x + v∂v/∂y = -∂p/∂y + (1/Re)∇²v
```

All derivatives computed by PyTorch autograd — no mesh, no discretisation.

---

## Why hemodynamics?

Wall Shear Stress (WSS) is the biomechanical link between fluid mechanics and vascular disease:

- **Low WSS (< 0.4 Pa)** → endothelial dysfunction, monocyte adhesion, plaque initiation
- **Oscillating WSS** → worse than uniformly low WSS  
- **The carotid bifurcation** is the canonical low-WSS site → most common location for atherosclerosis → leading cause of ischaemic stroke

PINNs can map WSS in complex geometries without meshing — in minutes on a free GPU.

---

## Requirements

```bash
pip install torch numpy matplotlib
```

All notebooks run on **free Google Colab GPU** (T4). Typical runtimes:
- Notebook 01: ~8 min
- Notebook 02: ~5 min  
- Notebook 03: ~15 min

---

## Notebook 01 — Lid-Driven Cavity

Classic benchmark problem. Unit square cavity, top wall sliding at u=1.

**Validation:** PINN centreline velocity compared against Ghia et al. (1982) tabulated data — the gold standard for Re=100.

The network takes (x,y) as input and outputs u, v, p simultaneously. No labeled training data — only PDE residuals and boundary conditions.

---

## Notebook 02 — Poiseuille Flow (Straight Blood Vessel)

Pressure-driven flow in a rectangular channel — the simplest model of a straight artery.

**Why it matters:** Has an exact analytical solution (parabolic profile), making it a perfect validation case before attempting complex geometries. Also introduces Wall Shear Stress computation.

**WSS formula:** τ_w = μ × du/dy |_{wall}

---

## Notebook 03 — Carotid Artery Bifurcation

2D Y-shaped bifurcation: Common Carotid → Internal Carotid + External Carotid.

**Key outputs:**
- Velocity field through the bifurcation
- Pressure drop from CCA to ICA/ECA
- WSS map identifying atherogenic (low-WSS) zones

**Clinical relevance:** The outer wall of the ICA sinus is where plaques preferentially form — driven by low, oscillating WSS. This is computable without a mesh, in 15 minutes.

---

## Architecture

All notebooks use the same core PINN:

```python
class PINN(nn.Module):
    # Input:  (x, y) — spatial coordinates
    # Output: (u, v, p) — velocity + pressure
    # Hidden: 4 × 64 or 4 × 128 neurons, tanh activation
    # Init:   Xavier normal
```

Training strategy: **Adam (20k epochs) → L-BFGS (10 steps)** — fast convergence + tight PDE satisfaction.

---

## References

- Raissi, Perdikaris, Karniadakis (2019). *Physics-informed neural networks.* JCP. [arXiv:1711.10561](https://arxiv.org/abs/1711.10561)
- Ghia, Ghia, Shin (1982). *High-Re solutions for incompressible flow using the Navier-Stokes equations.* JCP 48(3).
- Ku, Giddens, Zarins, Glagov (1985). *Pulsatile flow and atherosclerosis in the human carotid bifurcation.* Arteriosclerosis 5(3).
- Kissas et al. (2020). *Machine learning in cardiovascular flows modeling.* [arXiv:1905.11277](https://arxiv.org/abs/1905.11277)

---

## Author

**Prahlad G. Menon, PhD, PMP**  
Carnegie Mellon University — Biomedical Engineering  
[QuantMD](https://menonpg.github.io/quantmd-cv/) · [ThinkCreate.AI](https://thinkcreateai.com) · [Blog](https://blog.themenonlab.com)

MIT License
