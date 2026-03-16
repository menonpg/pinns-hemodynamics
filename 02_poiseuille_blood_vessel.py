# %% [markdown]
# # PINN 02 — Poiseuille Flow in a Blood Vessel
#
# **Physics context:** Blood in large vessels (aorta, carotid, femoral) is
# approximately Newtonian at physiological shear rates. Straight-vessel flow
# under a constant pressure gradient produces a **parabolic velocity profile**
# — Poiseuille flow — which has an exact analytical solution we can validate against.
#
# This notebook serves as the stepping stone before the carotid bifurcation.
# It also introduces **Wall Shear Stress (WSS)** — the biomechanical quantity
# most directly linked to endothelial dysfunction and atherosclerosis.
#
# **Geometry:** Rectangular channel, width 2H (vessel lumen), length L.
# Inlet: parabolic u-profile. Outlet: zero pressure. Walls: no-slip.
#
# **Analytical solution:**
# u(y) = U_max × (1 − (y/H)²)
# where U_max = (G × H²) / (2μ),  G = pressure gradient (Pa/m)

# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

torch.manual_seed(42)
np.random.seed(42)

# --- Geometry (normalised units) ---
L  = 4.0    # channel length  (x ∈ [0, L])
H  = 1.0    # half-width      (y ∈ [-H, H])

# --- Flow parameters ---
Re = 100.0              # Reynolds number (Re = U_max * 2H / ν)
U_max = 1.0             # peak centreline velocity (non-dimensional)
# Derived: pressure gradient G = 2 * U_max * μ / H²
# In non-dimensional NS: ∂p/∂x = -2/Re  (with ν=1/Re, H=1, U_max=1)

# %% [markdown]
# ## Analytical Solution (Ground Truth)

# %%
def analytical_u(y):
    """Poiseuille parabolic profile: u(y) = U_max(1 - (y/H)²)"""
    return U_max * (1.0 - (y / H)**2)

def analytical_wss(mu=1.0/Re):
    """Wall shear stress τ = μ du/dy|_{y=±H} = ∓ 2 μ U_max / H"""
    return 2.0 * mu * U_max / H

print(f"Peak velocity    : U_max = {U_max:.3f}")
print(f"Wall shear stress: τ_w  = {analytical_wss():.4f}  (at y = ±H)")

# %% [markdown]
# ## Neural Network

# %%
class PINN(nn.Module):
    def __init__(self, layers=[2, 64, 64, 64, 3]):
        super().__init__()
        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                net.append(nn.Tanh())
        self.net = nn.Sequential(*net)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]  # u, v, p

model = PINN().to(device)

# %% [markdown]
# ## Sampling

# %%
def sample_interior(n=6000):
    x = torch.rand(n, 1, device=device) * L
    y = (torch.rand(n, 1, device=device) * 2 - 1) * H
    return x, y

def sample_inlet(n=300):
    """x=0: parabolic u-profile, v=0"""
    x = torch.zeros(n, 1, device=device)
    y = (torch.rand(n, 1, device=device) * 2 - 1) * H
    u_in = torch.tensor(analytical_u(y.cpu().numpy()), dtype=torch.float32, device=device)
    return x, y, u_in

def sample_walls(n=600):
    """y=±H: no-slip, u=v=0"""
    x_top = torch.rand(n, 1, device=device) * L
    y_top = torch.full((n, 1), H, device=device)
    x_bot = torch.rand(n, 1, device=device) * L
    y_bot = torch.full((n, 1), -H, device=device)
    return (x_top, y_top), (x_bot, y_bot)

def sample_outlet(n=300):
    """x=L: zero pressure"""
    x = torch.full((n, 1), L, device=device)
    y = (torch.rand(n, 1, device=device) * 2 - 1) * H
    return x, y

x_int, y_int = sample_interior()
x_in, y_in, u_in_target = sample_inlet()
(x_top, y_top), (x_bot, y_bot) = sample_walls()
x_out, y_out = sample_outlet()

# %% [markdown]
# ## PDE Residuals (Stokes / Navier-Stokes)

# %%
def ns_residuals(model, x, y):
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

    continuity = u_x + v_y
    x_mom = u * u_x + v * u_y + p_x - (1.0/Re) * (u_xx + u_yy)
    y_mom = u * v_x + v * v_y + p_y - (1.0/Re) * (v_xx + v_yy)
    return continuity, x_mom, y_mom

# %% [markdown]
# ## Loss and Training

# %%
def compute_loss(model):
    cont, xm, ym = ns_residuals(model, x_int, y_int)
    loss_pde = (cont**2).mean() + (xm**2).mean() + (ym**2).mean()

    # Inlet: parabolic u, v=0
    u_i, v_i, _ = model(x_in, y_in)
    loss_bc = ((u_i - u_in_target)**2).mean() + (v_i**2).mean()

    # Walls: no-slip
    u_t, v_t, _ = model(x_top, y_top)
    u_b, v_b, _ = model(x_bot, y_bot)
    loss_bc += (u_t**2).mean() + (v_t**2).mean()
    loss_bc += (u_b**2).mean() + (v_b**2).mean()

    # Outlet: p=0
    _, _, p_out = model(x_out, y_out)
    loss_bc += (p_out**2).mean()

    return loss_pde + loss_bc

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []

print("Training...")
for epoch in range(15000):
    optimizer.zero_grad()
    loss = compute_loss(model)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 3000 == 0:
        print(f"  Epoch {epoch:6d} | Loss {loss.item():.4e}")

# L-BFGS refinement
lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=50,
                            line_search_fn="strong_wolfe")
def closure():
    lbfgs.zero_grad(); l = compute_loss(model); l.backward(); return l
for _ in range(10):
    lbfgs.step(closure)
print(f"Final loss: {compute_loss(model).item():.4e}")

# %% [markdown]
# ## Wall Shear Stress (WSS)
#
# **Clinical significance:** In straight vessels, WSS ≈ 1.5–2.5 Pa is protective.
# Low WSS (< 0.4 Pa) promotes inflammatory gene expression, monocyte adhesion,
# and plaque initiation. This is why bifurcations are dangerous — the outer wall
# of a bend sees low, oscillating WSS.
#
# τ_w = μ × ∂u/∂y |_{y=±H}

# %%
model.eval()
n_wss = 200
x_wss = torch.linspace(0, L, n_wss, device=device).unsqueeze(1)

# Top wall WSS
y_wss_top = torch.full((n_wss, 1), H, device=device)
x_wss_g = x_wss.clone().requires_grad_(True)
y_wss_g = y_wss_top.clone().requires_grad_(True)
u_top, _, _ = model(x_wss_g, y_wss_g)
du_dy_top = torch.autograd.grad(u_top.sum(), y_wss_g, create_graph=False)[0]
wss_top = -(1.0/Re) * du_dy_top.detach().cpu().numpy()  # τ = -μ du/dy at top

# Analytical WSS
wss_analytical = analytical_wss()
print(f"PINN WSS (mean, top wall) : {abs(wss_top.mean()):.4f}")
print(f"Analytical WSS            : {wss_analytical:.4f}")

# %% [markdown]
# ## Visualisation

# %%
model.eval()
with torch.no_grad():
    nx, ny = 80, 40
    xs = torch.linspace(0, L, nx, device=device)
    ys = torch.linspace(-H, H, ny, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    x_flat = X.reshape(-1, 1); y_flat = Y.reshape(-1, 1)
    U, V, P = model(x_flat, y_flat)
    U = U.cpu().numpy().reshape(ny, nx)
    V = V.cpu().numpy().reshape(ny, nx)
    P = P.cpu().numpy().reshape(ny, nx)
    X_np = X.cpu().numpy(); Y_np = Y.cpu().numpy()

# u profile at x=L/2 vs analytical
x_mid = torch.full((100, 1), L/2, device=device)
y_prof = torch.linspace(-H, H, 100, device=device).unsqueeze(1)
with torch.no_grad():
    u_pred, _, _ = model(x_mid, y_prof)
u_pred = u_pred.cpu().numpy().squeeze()
y_prof_np = y_prof.cpu().numpy().squeeze()
u_anal = analytical_u(y_prof_np)

fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#0d0d0d')
fig.suptitle('PINN — Poiseuille Flow (Blood Vessel, Re=100)',
             color='white', fontsize=14, fontweight='bold')

kw = dict(facecolor='#1a1a1a')

ax = axes[0, 0]
ax.set(**kw)
cf = ax.contourf(X_np, Y_np, U, levels=50, cmap='inferno')
ax.streamplot(X_np, Y_np, U, V, color='white', linewidth=0.5, density=1.0)
plt.colorbar(cf, ax=ax, label='u-velocity')
ax.set_title('u-Velocity + Streamlines', color='white')
ax.set_xlabel('x', color='white'); ax.set_ylabel('y', color='white')
ax.axhline(H, color='#ef4444', lw=1.5, label='vessel wall')
ax.axhline(-H, color='#ef4444', lw=1.5)
ax.tick_params(colors='white')
for s in ax.spines.values(): s.set_edgecolor('#444')

ax = axes[0, 1]
ax.set(**kw)
ax.plot(u_pred, y_prof_np, color='#6366f1', lw=2.5, label='PINN')
ax.plot(u_anal, y_prof_np, color='#f97316', lw=2, linestyle='--', label='Analytical')
ax.set_xlabel('u-velocity', color='white'); ax.set_ylabel('y', color='white')
ax.set_title('Velocity Profile at x=L/2\nPINN vs Analytical', color='white')
ax.legend(facecolor='#2a2a2a', edgecolor='#444', labelcolor='white')
ax.tick_params(colors='white')
for s in ax.spines.values(): s.set_edgecolor('#444')

ax = axes[1, 0]
ax.set(**kw)
x_wss_np = x_wss.cpu().numpy().squeeze()
ax.plot(x_wss_np, abs(wss_top), color='#22c55e', lw=2, label='PINN WSS (top wall)')
ax.axhline(wss_analytical, color='#f97316', lw=2, linestyle='--', label=f'Analytical = {wss_analytical:.4f}')
ax.set_xlabel('x (along vessel)', color='white'); ax.set_ylabel('|τ_w|', color='white')
ax.set_title('Wall Shear Stress\nClinical range: 1.5–2.5 Pa protective', color='white')
ax.legend(facecolor='#2a2a2a', edgecolor='#444', labelcolor='white')
ax.tick_params(colors='white')
for s in ax.spines.values(): s.set_edgecolor('#444')

ax = axes[1, 1]
ax.set(**kw)
ax.semilogy(losses, color='#22c55e', lw=1.5)
ax.set_xlabel('Epoch', color='white'); ax.set_ylabel('Loss', color='white')
ax.set_title('Training Loss', color='white')
ax.tick_params(colors='white')
for s in ax.spines.values(): s.set_edgecolor('#444')

plt.tight_layout()
plt.savefig('poiseuille_blood_vessel.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: poiseuille_blood_vessel.png")
