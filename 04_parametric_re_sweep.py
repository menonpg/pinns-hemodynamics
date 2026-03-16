# %% [markdown]
# # PINN 04 — Parametric Re Sweep: Lid-Driven Cavity
#
# **The practical question:** I've trained a PINN at Re=100. Now what?
# Can I change Re without retraining from scratch?
#
# **Answer: parametric PINN.** Add Re as a third input to the network.
# Train on Re ∈ {100, 400, 1000} simultaneously. At inference, query any Re —
# including values the model was never trained on — and get a physics-consistent
# flow field instantly.
#
# **What you'll see as Re increases:**
# - Re=100:  Single smooth vortex, slightly above centre
# - Re=400:  Vortex migrates toward centre, corner vortices begin forming
# - Re=1000: Vortex near centre, bottom-right secondary vortex clearly visible
# - Re>3200: Flow becomes unsteady — steady PINN assumption breaks down here
#
# **Why this matters for hemodynamics:**
# Blood vessels operate at Re ≈ 50–500 (arterioles → carotid). The same
# parametric approach lets you map WSS vs Re for a patient-specific geometry
# without re-solving for each Re individually — directly relevant for
# exercise-induced flow changes, stenosis progression, or drug effects.
#
# **Benchmark:** Ghia et al. (1982), Re=100/400/1000 centreline profiles.
#
# Runs in ~20 min on free Colab GPU.

# %% [markdown]
# ## Setup

# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
torch.manual_seed(42); np.random.seed(42)

# Reynolds numbers to train on
RE_LIST   = [100.0, 400.0, 1000.0]
RE_MIN    = 100.0
RE_MAX    = 1000.0

def normalise_re(re):
    """Map Re ∈ [100, 1000] → [0, 1] for network input."""
    return (re - RE_MIN) / (RE_MAX - RE_MIN)

# %% [markdown]
# ## Parametric Network
#
# The only change from a standard PINN: input is now **(x, y, Re_norm)**.
# Everything else — architecture, loss, training — is identical.
# The network has to learn that higher Re means stronger inertia, weaker viscous damping.

# %%
class ParametricPINN(nn.Module):
    def __init__(self, layers=[3, 64, 64, 64, 64, 3]):   # 3 inputs now
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

    def forward(self, x, y, re_norm):
        inp = torch.cat([x, y, re_norm], dim=1)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]   # u, v, p

model = ParametricPINN().to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
# ## Sampling Strategy
#
# For each batch: sample interior and boundary points for ALL Re values.
# The model sees all three Re values every iteration — this is what
# forces it to learn the Re-dependent physics, not just memorise one solution.

# %%
def make_re_tensor(re_val, n):
    """Create a column tensor of normalised Re value, shape (n,1)."""
    re_norm = normalise_re(re_val)
    return torch.full((n, 1), re_norm, device=device)

def sample_interior(n=3000):
    x = torch.rand(n, 1, device=device)
    y = torch.rand(n, 1, device=device)
    return x, y

def sample_boundaries(n_per_wall=300):
    x_b = torch.rand(n_per_wall, 1, device=device); y_b = torch.zeros(n_per_wall, 1, device=device)
    x_t = torch.rand(n_per_wall, 1, device=device); y_t = torch.ones(n_per_wall, 1, device=device)
    x_l = torch.zeros(n_per_wall, 1, device=device); y_l = torch.rand(n_per_wall, 1, device=device)
    x_r = torch.ones(n_per_wall, 1, device=device);  y_r = torch.rand(n_per_wall, 1, device=device)
    return (x_b, y_b), (x_t, y_t), (x_l, y_l), (x_r, y_r)

# Pre-sample once
x_int, y_int = sample_interior(3000)
(x_b, y_b), (x_t, y_t), (x_l, y_l), (x_r, y_r) = sample_boundaries(300)

# %% [markdown]
# ## PDE Residuals (same as before, but Re is passed as a parameter)

# %%
def ns_residuals(model, x, y, re_val):
    """Navier-Stokes residuals at physical Reynolds number re_val."""
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    re_norm = make_re_tensor(re_val, x.shape[0])

    u, v, p = model(x, y, re_norm)

    def grad(f, var):
        return torch.autograd.grad(f, var,
                                   grad_outputs=torch.ones_like(f),
                                   create_graph=True)[0]

    u_x = grad(u, x); u_y = grad(u, y)
    v_x = grad(v, x); v_y = grad(v, y)
    p_x = grad(p, x); p_y = grad(p, y)
    u_xx = grad(u_x, x); u_yy = grad(u_y, y)
    v_xx = grad(v_x, x); v_yy = grad(v_y, y)

    cont = u_x + v_y
    xmom = u*u_x + v*u_y + p_x - (1.0/re_val)*(u_xx + u_yy)
    ymom = u*v_x + v*v_y + p_y - (1.0/re_val)*(v_xx + v_yy)
    return cont, xmom, ymom

# %% [markdown]
# ## Loss Function
#
# Loss is summed over ALL Re values in each iteration.
# This is what forces the network to interpolate between Re values correctly.

# %%
def compute_loss(model):
    total_loss = 0.0

    for re_val in RE_LIST:
        re_norm = make_re_tensor(re_val, x_int.shape[0])

        # PDE residuals
        cont, xm, ym = ns_residuals(model, x_int, y_int, re_val)
        loss_pde = (cont**2).mean() + (xm**2).mean() + (ym**2).mean()

        # Boundary conditions — same for all Re
        re_b = make_re_tensor(re_val, x_b.shape[0])
        re_t = make_re_tensor(re_val, x_t.shape[0])
        re_l = make_re_tensor(re_val, x_l.shape[0])
        re_r = make_re_tensor(re_val, x_r.shape[0])

        u_b_, v_b_, _ = model(x_b, y_b, re_b)
        u_t_, v_t_, _ = model(x_t, y_t, re_t)
        u_l_, v_l_, _ = model(x_l, y_l, re_l)
        u_r_, v_r_, _ = model(x_r, y_r, re_r)

        loss_bc = ((u_b_**2).mean() + (v_b_**2).mean() +
                   ((u_t_ - 1)**2).mean() + (v_t_**2).mean() +
                   (u_l_**2).mean() + (v_l_**2).mean() +
                   (u_r_**2).mean() + (v_r_**2).mean())

        # Pressure pin at (0,0)
        x0 = torch.zeros(1,1,device=device); y0 = torch.zeros(1,1,device=device)
        re_0 = make_re_tensor(re_val, 1)
        _, _, p0 = model(x0, y0, re_0)
        loss_p = p0**2

        total_loss = total_loss + loss_pde + loss_bc + loss_p

    return total_loss

# %% [markdown]
# ## Training

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []

print("Phase 1: Adam (training on Re=100, 400, 1000 simultaneously)")
for epoch in range(25000):
    optimizer.zero_grad()
    loss = compute_loss(model)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 5000 == 0:
        print(f"  Epoch {epoch:6d} | Loss {loss.item():.4e}")

# %%
print("\nPhase 2: L-BFGS refinement")
lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=50,
                            line_search_fn="strong_wolfe")
def closure():
    lbfgs.zero_grad(); l = compute_loss(model); l.backward(); return l
for step in range(10):
    lbfgs.step(closure)
print(f"Final loss: {compute_loss(model).item():.4e}")

# %% [markdown]
# ## Ghia 1982 Benchmark Data
#
# u-velocity along vertical centreline (x=0.5) for Re=100, 400, 1000.
# Source: Ghia, Ghia, Shin (1982), Journal of Computational Physics 48(3).

# %%
ghia_y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516,
                   0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719,
                   0.1016, 0.0703, 0.0625, 0.0547, 0.0000])

ghia_u = {
    100:  np.array([ 1.0000,  0.8412,  0.7887,  0.7372,  0.6872,  0.2336,
                     0.0033, -0.1364, -0.2058, -0.2109, -0.1566, -0.1015,
                    -0.0643, -0.0478, -0.0419, -0.0330,  0.0000]),
    400:  np.array([ 1.0000,  0.7576,  0.7147,  0.6736,  0.6303,  0.2928,
                     0.0919, -0.0889, -0.2102, -0.2173, -0.1753, -0.1490,
                    -0.1017, -0.0731, -0.0642, -0.0535,  0.0000]),
    1000: np.array([ 1.0000,  0.6560,  0.5948,  0.5353,  0.4764,  0.3383,
                     0.1685,  0.0170, -0.0619, -0.0945, -0.2440, -0.2803,
                    -0.2767, -0.2434, -0.2208, -0.1877,  0.0000]),
}

# %% [markdown]
# ## Validate: PINN vs Ghia at All Three Re Values

# %%
model.eval()
n_cl = 200
y_cl = torch.linspace(0, 1, n_cl, device=device).unsqueeze(1)
x_cl = torch.full_like(y_cl, 0.5)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0d0d0d')
colours = ['#6366f1', '#22c55e', '#f97316']

for i, re_val in enumerate(RE_LIST):
    ax = axes[i]; ax.set_facecolor('#1a1a1a')
    re_norm_cl = make_re_tensor(re_val, n_cl)
    with torch.no_grad():
        u_pred, _, _ = model(x_cl, y_cl, re_norm_cl)
    u_pred_np = u_pred.cpu().numpy().squeeze()
    y_cl_np   = y_cl.cpu().numpy().squeeze()

    ax.plot(u_pred_np, y_cl_np, color=colours[i], lw=2.5, label='PINN')
    ax.scatter(ghia_u[int(re_val)], ghia_y, color='white', s=50,
               zorder=5, label='Ghia 1982')
    ax.axvline(0, color='white', lw=0.4, alpha=0.3)
    ax.set_title(f'Re = {int(re_val)}', color='white', fontsize=13, fontweight='bold')
    ax.set_xlabel('u-velocity', color='white'); ax.set_ylabel('y', color='white')
    ax.legend(facecolor='#2a2a2a', edgecolor='#444', labelcolor='white')
    ax.tick_params(colors='white')
    for s in ax.spines.values(): s.set_edgecolor('#444')

fig.suptitle('Parametric PINN — Centreline Validation vs Ghia 1982\nSingle network, three Reynolds numbers',
             color='white', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('validation_ghia_all_re.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()

# %% [markdown]
# ## Re Sweep: Watch the Flow Evolve
#
# Now query the trained model at Re values it was NEVER trained on.
# This is the key capability — parametric inference, not re-solving.

# %%
re_sweep = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
n = 80

fig = plt.figure(figsize=(20, 8), facecolor='#0d0d0d')
fig.suptitle('Re Sweep: 100 → 1000\n(Single parametric network — no retraining)',
             color='white', fontsize=14, fontweight='bold')

for idx, re_val in enumerate(re_sweep):
    ax = fig.add_subplot(2, 5, idx+1, facecolor='#1a1a1a')

    with torch.no_grad():
        xs = torch.linspace(0, 1, n, device=device)
        ys = torch.linspace(0, 1, n, device=device)
        X, Y = torch.meshgrid(xs, ys, indexing='xy')
        x_f = X.reshape(-1,1); y_f = Y.reshape(-1,1)
        re_n = make_re_tensor(re_val, x_f.shape[0])
        U, V, _ = model(x_f, y_f, re_n)
        U = U.cpu().numpy().reshape(n, n)
        V = V.cpu().numpy().reshape(n, n)
        speed = np.sqrt(U**2 + V**2)
        X_np = X.cpu().numpy(); Y_np = Y.cpu().numpy()

    ax.contourf(X_np, Y_np, speed, levels=30, cmap='inferno', vmin=0, vmax=0.7)
    try:
        ax.streamplot(X_np, Y_np, U, V, color='white', linewidth=0.4,
                      density=1.0, arrowsize=0.5)
    except Exception:
        pass
    ax.set_title(f'Re={re_val}', color='white', fontsize=10, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor('#333')

plt.tight_layout()
plt.savefig('re_sweep_flow.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()

# %% [markdown]
# ## Vortex Centre Migration vs Re
#
# Track the primary vortex centre (the point of minimum speed) as Re increases.
# This is a quantitative summary of the Re-sweep in one plot.
# Known reference: Ghia primary vortex centre (x, y)
#   Re=100:  (0.617, 0.742)
#   Re=400:  (0.556, 0.606)
#   Re=1000: (0.533, 0.564)

# %%
ghia_vortex = {100: (0.617, 0.742), 400: (0.556, 0.606), 1000: (0.533, 0.564)}

re_vals_fine = np.linspace(100, 1000, 50)
vortex_x, vortex_y = [], []

n_v = 100
xs = torch.linspace(0, 1, n_v, device=device)
ys = torch.linspace(0, 1, n_v, device=device)
X, Y = torch.meshgrid(xs, ys, indexing='xy')
x_f = X.reshape(-1,1); y_f = Y.reshape(-1,1)

model.eval()
with torch.no_grad():
    for re_val in re_vals_fine:
        re_n = make_re_tensor(float(re_val), x_f.shape[0])
        U, V, _ = model(x_f, y_f, re_n)
        speed = (U**2 + V**2).cpu().numpy().reshape(n_v, n_v)
        idx_min = np.unravel_index(speed.argmin(), speed.shape)
        vortex_x.append(xs.cpu().numpy()[idx_min[1]])
        vortex_y.append(ys.cpu().numpy()[idx_min[0]])

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#0d0d0d')

ax = axes[0]; ax.set_facecolor('#1a1a1a')
sc = ax.scatter(vortex_x, vortex_y, c=re_vals_fine, cmap='plasma', s=40, zorder=3)
for re_ref, (gx, gy) in ghia_vortex.items():
    ax.scatter(gx, gy, marker='*', s=200, color='white', zorder=5)
    ax.annotate(f'Ghia Re={re_ref}', (gx, gy), textcoords='offset points',
                xytext=(6, 4), color='white', fontsize=8)
plt.colorbar(sc, ax=ax, label='Re')
ax.set_xlabel('Vortex centre x', color='white')
ax.set_ylabel('Vortex centre y', color='white')
ax.set_title('Primary Vortex Centre Migration\n(moves toward (0.5,0.5) as Re ↑)',
             color='white', fontsize=11)
ax.tick_params(colors='white')
for s in ax.spines.values(): s.set_edgecolor('#444')

ax = axes[1]; ax.set_facecolor('#1a1a1a')
ax.plot(re_vals_fine, vortex_x, color='#6366f1', lw=2, label='x centre (PINN)')
ax.plot(re_vals_fine, vortex_y, color='#22c55e', lw=2, label='y centre (PINN)')
for re_ref, (gx, gy) in ghia_vortex.items():
    ax.scatter(re_ref, gx, color='#6366f1', marker='*', s=150, zorder=5)
    ax.scatter(re_ref, gy, color='#22c55e', marker='*', s=150, zorder=5)
ax.set_xlabel('Re', color='white'); ax.set_ylabel('Vortex centre coordinate', color='white')
ax.set_title('Vortex Centre vs Re\n(stars = Ghia reference)', color='white', fontsize=11)
ax.legend(facecolor='#2a2a2a', edgecolor='#444', labelcolor='white')
ax.tick_params(colors='white')
for s in ax.spines.values(): s.set_edgecolor('#444')

plt.tight_layout()
plt.savefig('vortex_centre_vs_re.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()

# %% [markdown]
# ## Centreline Profile Sweep — Continuous Re
#
# Query the model at 10 Re values from 100→1000 and overlay the centreline
# u-profiles. Watch how the near-wall velocity gradient steepens (↑ WSS)
# and the reversal region deepens as Re increases.

# %%
re_profile_sweep = np.linspace(100, 1000, 10)
y_cl = torch.linspace(0, 1, 200, device=device).unsqueeze(1)
x_cl = torch.full_like(y_cl, 0.5)

fig, ax = plt.subplots(figsize=(9, 6), facecolor='#0d0d0d')
ax.set_facecolor('#1a1a1a')

cmap = plt.cm.plasma
norm = Normalize(vmin=100, vmax=1000)

model.eval()
with torch.no_grad():
    for re_val in re_profile_sweep:
        re_n = make_re_tensor(float(re_val), y_cl.shape[0])
        u_pred, _, _ = model(x_cl, y_cl, re_n)
        colour = cmap(norm(re_val))
        ax.plot(u_pred.cpu().numpy().squeeze(),
                y_cl.cpu().numpy().squeeze(),
                color=colour, lw=1.8, alpha=0.9,
                label=f'Re={int(re_val)}')

# Ghia reference dots
for re_ref in [100, 400, 1000]:
    ax.scatter(ghia_u[re_ref], ghia_y, marker='o', s=30,
               color=cmap(norm(re_ref)), zorder=5, alpha=0.7)

ax.axvline(0, color='white', lw=0.5, alpha=0.3)
plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Re')
ax.set_xlabel('u-velocity at x=0.5', color='white', fontsize=12)
ax.set_ylabel('y', color='white', fontsize=12)
ax.set_title('Centreline u-Profile: Re 100 → 1000\nDots = Ghia 1982 reference',
             color='white', fontsize=13, fontweight='bold')
ax.tick_params(colors='white')
for s in ax.spines.values(): s.set_edgecolor('#444')

plt.tight_layout()
plt.savefig('centreline_re_sweep.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()

# %% [markdown]
# ## What You're Seeing — Physical Interpretation
#
# As Re increases from 100 → 1000:
#
# **1. Vortex centre moves toward (0.5, 0.5)**
# At low Re, viscosity dominates and the vortex is pulled upward (toward the
# moving lid). At high Re, inertia dominates and the vortex settles centrally.
#
# **2. Velocity reversal deepens**
# The bottom half of the centreline profile shows increasingly negative u
# (flow returning leftward in the recirculation zone).
#
# **3. Corner vortices grow**
# At Re=1000 you'll see secondary vortices forming in the bottom corners —
# these are physically real and appear in FEM solutions at the same Re.
#
# **4. Near-wall gradient steepens**
# The velocity profile near the top lid becomes sharper at high Re —
# higher wall shear stress, steeper boundary layer.
#
# **5. Why Re>3200 fails here**
# The steady Navier-Stokes assumption breaks down. Real flow becomes
# time-periodic (Hopf bifurcation ~Re=7400 for LDC). A PINN for
# unsteady flow needs time t as an additional input.
#
# **Hemodynamics connection:**
# Blood velocity in the carotid artery changes with exercise (Re 200 → 500).
# This parametric approach lets you map WSS vs exercise intensity for a
# patient-specific geometry — without resolving at each Re separately.

print("\n✅ Parametric Re sweep complete.")
print("Key insight: one network, any Re in [100,1000], instant inference.")
print("This is the foundation of physics-informed surrogate modelling.")
