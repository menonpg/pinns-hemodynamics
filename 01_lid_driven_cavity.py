# %% [markdown]
# # PINN 01 — Lid-Driven Cavity (Re=100)
#
# **Physics-Informed Neural Network for 2D incompressible Navier-Stokes**
#
# A unit square cavity with the top wall sliding at u=1. No mesh, no finite
# differences — the neural network is trained to satisfy the NS equations
# everywhere by minimising PDE residuals via automatic differentiation.
#
# **Benchmark:** Ghia et al. (1982) centreline velocity profiles at Re=100.
#
# Runs in ~10 min on free Colab GPU.

# %% [markdown]
# ## Setup

# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

torch.manual_seed(42)
np.random.seed(42)

Re = 100.0   # Reynolds number

# %% [markdown]
# ## Neural Network Architecture
#
# Input:  (x, y) ∈ [0,1]²
# Output: (u, v, p) — x-velocity, y-velocity, pressure
#
# Tanh activation is essential — we need smooth second derivatives for
# the viscous terms in Navier-Stokes.

# %%
class PINN(nn.Module):
    def __init__(self, layers=[2, 64, 64, 64, 64, 3]):
        super().__init__()
        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                net.append(nn.Tanh())
        self.net = nn.Sequential(*net)

        # Xavier initialisation — helps with tanh saturation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        out = self.net(xy)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]   # u, v, p

model = PINN().to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
# ## Sampling Collocation and Boundary Points
#
# PINNs don't use a mesh — we sample random points inside the domain
# (collocation points) and on each boundary.

# %%
def sample_interior(n=8000):
    x = torch.rand(n, 1, device=device)
    y = torch.rand(n, 1, device=device)
    return x, y

def sample_boundaries(n_per_wall=400):
    # Bottom wall: y=0, u=v=0
    x_b = torch.rand(n_per_wall, 1, device=device)
    y_b = torch.zeros(n_per_wall, 1, device=device)

    # Top wall: y=1, u=1, v=0  (lid moving right)
    x_t = torch.rand(n_per_wall, 1, device=device)
    y_t = torch.ones(n_per_wall, 1, device=device)

    # Left wall: x=0, u=v=0
    x_l = torch.zeros(n_per_wall, 1, device=device)
    y_l = torch.rand(n_per_wall, 1, device=device)

    # Right wall: x=1, u=v=0
    x_r = torch.ones(n_per_wall, 1, device=device)
    y_r = torch.rand(n_per_wall, 1, device=device)

    return (x_b, y_b), (x_t, y_t), (x_l, y_l), (x_r, y_r)

# Sample once — fixed collocation points throughout training
x_int, y_int = sample_interior(8000)
(x_b, y_b), (x_t, y_t), (x_l, y_l), (x_r, y_r) = sample_boundaries(400)

# %% [markdown]
# ## PDE Residuals via Automatic Differentiation
#
# The incompressible Navier-Stokes equations:
#
# **Continuity (mass conservation):**
# ∂u/∂x + ∂v/∂y = 0
#
# **x-Momentum:**
# u∂u/∂x + v∂u/∂y = −∂p/∂x + (1/Re)(∂²u/∂x² + ∂²u/∂y²)
#
# **y-Momentum:**
# u∂v/∂x + v∂v/∂y = −∂p/∂y + (1/Re)(∂²v/∂x² + ∂²v/∂y²)
#
# PyTorch autograd computes all derivatives analytically — no finite differences.

# %%
def ns_residuals(model, x, y):
    """Compute Navier-Stokes PDE residuals at (x,y) collocation points."""
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)

    u, v, p = model(x, y)

    def grad(f, var):
        return torch.autograd.grad(f, var,
                                   grad_outputs=torch.ones_like(f),
                                   create_graph=True)[0]

    u_x = grad(u, x);  u_y = grad(u, y)
    v_x = grad(v, x);  v_y = grad(v, y)
    p_x = grad(p, x);  p_y = grad(p, y)

    u_xx = grad(u_x, x);  u_yy = grad(u_y, y)
    v_xx = grad(v_x, x);  v_yy = grad(v_y, y)

    continuity  = u_x + v_y
    x_momentum  = u * u_x + v * u_y + p_x - (1.0/Re) * (u_xx + u_yy)
    y_momentum  = u * v_x + v * v_y + p_y - (1.0/Re) * (v_xx + v_yy)

    return continuity, x_momentum, y_momentum

# %% [markdown]
# ## Loss Function
#
# Total loss = PDE residuals + boundary conditions
#
# Each term is the mean squared error of the violation.

# %%
def compute_loss(model):
    # --- Interior: PDE residuals ---
    cont, xmom, ymom = ns_residuals(model, x_int, y_int)
    loss_pde = (cont**2).mean() + (xmom**2).mean() + (ymom**2).mean()

    # --- Boundary conditions ---
    # Bottom wall: u=v=0
    u_b, v_b, _ = model(x_b, y_b)
    loss_bc  = (u_b**2).mean() + (v_b**2).mean()

    # Top lid: u=1, v=0
    u_t, v_t, _ = model(x_t, y_t)
    loss_bc += ((u_t - 1)**2).mean() + (v_t**2).mean()

    # Left wall: u=v=0
    u_l, v_l, _ = model(x_l, y_l)
    loss_bc += (u_l**2).mean() + (v_l**2).mean()

    # Right wall: u=v=0
    u_r, v_r, _ = model(x_r, y_r)
    loss_bc += (u_r**2).mean() + (v_r**2).mean()

    # Pressure pin: p=0 at (0,0) to remove gauge freedom
    x0 = torch.zeros(1, 1, device=device)
    y0 = torch.zeros(1, 1, device=device)
    _, _, p0 = model(x0, y0)
    loss_p = p0**2

    return loss_pde + loss_bc + loss_p

# %% [markdown]
# ## Training
#
# Two-phase training:
# 1. **Adam** — fast initial convergence
# 2. **L-BFGS** — second-order refinement for tight PDE satisfaction

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=5000, gamma=0.5)

losses = []
print("Phase 1: Adam optimisation")
for epoch in range(20000):
    optimizer.zero_grad()
    loss = compute_loss(model)
    loss.backward()
    optimizer.step()
    scheduler.step()
    losses.append(loss.item())

    if epoch % 2000 == 0:
        print(f"  Epoch {epoch:6d} | Loss: {loss.item():.4e}")

# %%
print("\nPhase 2: L-BFGS refinement")
lbfgs = torch.optim.LBFGS(model.parameters(),
                            lr=0.1,
                            max_iter=50,
                            history_size=50,
                            line_search_fn="strong_wolfe")

def closure():
    lbfgs.zero_grad()
    loss = compute_loss(model)
    loss.backward()
    return loss

for step in range(10):
    lbfgs.step(closure)
    loss = compute_loss(model)
    losses.append(loss.item())
    print(f"  L-BFGS step {step+1:3d} | Loss: {loss.item():.4e}")

# %% [markdown]
# ## Visualisation

# %%
model.eval()
with torch.no_grad():
    n = 100
    x_vis = torch.linspace(0, 1, n, device=device)
    y_vis = torch.linspace(0, 1, n, device=device)
    X, Y = torch.meshgrid(x_vis, y_vis, indexing='xy')
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)

    U, V, P = model(x_flat, y_flat)
    U = U.cpu().numpy().reshape(n, n)
    V = V.cpu().numpy().reshape(n, n)
    P = P.cpu().numpy().reshape(n, n)
    speed = np.sqrt(U**2 + V**2)
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()

# %% [markdown]
# ## Ghia et al. (1982) Benchmark Comparison
#
# Gold standard reference data for Re=100 lid-driven cavity.
# We compare the u-velocity profile along the vertical centreline (x=0.5).

# %%
# Ghia 1982 — Re=100, u-velocity along vertical centreline (x=0.5)
ghia_y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516,
                   0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719,
                   0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
ghia_u = np.array([ 1.0000,  0.8412,  0.7887,  0.7372,  0.6872,  0.2336,
                    0.0033, -0.1364, -0.2058, -0.2109, -0.1566, -0.1015,
                   -0.0643, -0.0478, -0.0419, -0.0330,  0.0000])

# PINN prediction along x=0.5
with torch.no_grad():
    x_cl = torch.full((n, 1), 0.5, device=device)
    y_cl = torch.linspace(0, 1, n, device=device).unsqueeze(1)
    u_cl, _, _ = model(x_cl, y_cl)
    u_cl = u_cl.cpu().numpy().squeeze()
    y_cl = y_cl.cpu().numpy().squeeze()

# %% [markdown]
# ## Plots

# %%
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0d0d0d')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

kw_ax = dict(facecolor='#1a1a1a')

# 1. Velocity magnitude + streamlines
ax1 = fig.add_subplot(gs[0, 0], **kw_ax)
cf = ax1.contourf(X_np, Y_np, speed, levels=50, cmap='inferno')
ax1.streamplot(X_np, Y_np, U, V, color='white', linewidth=0.5, density=1.5, arrowsize=0.8)
plt.colorbar(cf, ax=ax1, label='|u|')
ax1.set_title('Velocity Magnitude + Streamlines', color='white', fontsize=11)
ax1.set_xlabel('x', color='white'); ax1.set_ylabel('y', color='white')
ax1.tick_params(colors='white')

# 2. Pressure field
ax2 = fig.add_subplot(gs[0, 1], **kw_ax)
cf2 = ax2.contourf(X_np, Y_np, P, levels=50, cmap='RdBu_r')
plt.colorbar(cf2, ax=ax2, label='p')
ax2.set_title('Pressure Field', color='white', fontsize=11)
ax2.set_xlabel('x', color='white'); ax2.set_ylabel('y', color='white')
ax2.tick_params(colors='white')

# 3. U-velocity contours
ax3 = fig.add_subplot(gs[0, 2], **kw_ax)
cf3 = ax3.contourf(X_np, Y_np, U, levels=50, cmap='coolwarm')
plt.colorbar(cf3, ax=ax3, label='u')
ax3.set_title('u-Velocity Contours', color='white', fontsize=11)
ax3.set_xlabel('x', color='white'); ax3.set_ylabel('y', color='white')
ax3.tick_params(colors='white')

# 4. Ghia benchmark comparison
ax4 = fig.add_subplot(gs[1, 0], **kw_ax)
ax4.plot(u_cl, y_cl, color='#6366f1', lw=2.5, label='PINN')
ax4.scatter(ghia_u, ghia_y, color='#f97316', s=60, zorder=5, label='Ghia et al. (1982)')
ax4.axvline(0, color='white', lw=0.5, alpha=0.3)
ax4.set_xlabel('u-velocity', color='white'); ax4.set_ylabel('y', color='white')
ax4.set_title('Centreline u-velocity (x=0.5)\nvs Ghia 1982 benchmark', color='white', fontsize=11)
ax4.legend(facecolor='#2a2a2a', edgecolor='#444', labelcolor='white', fontsize=9)
ax4.tick_params(colors='white')

# 5. V-velocity contours
ax5 = fig.add_subplot(gs[1, 1], **kw_ax)
cf5 = ax5.contourf(X_np, Y_np, V, levels=50, cmap='coolwarm')
plt.colorbar(cf5, ax=ax5, label='v')
ax5.set_title('v-Velocity Contours', color='white', fontsize=11)
ax5.set_xlabel('x', color='white'); ax5.set_ylabel('y', color='white')
ax5.tick_params(colors='white')

# 6. Training loss
ax6 = fig.add_subplot(gs[1, 2], **kw_ax)
ax6.semilogy(losses, color='#22c55e', lw=1.5)
ax6.set_xlabel('Epoch', color='white'); ax6.set_ylabel('Loss', color='white')
ax6.set_title('Training Loss', color='white', fontsize=11)
ax6.tick_params(colors='white')
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

fig.suptitle('PINN — Lid-Driven Cavity Re=100', color='white', fontsize=15, fontweight='bold')
plt.savefig('lid_driven_cavity_re100.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: lid_driven_cavity_re100.png")
