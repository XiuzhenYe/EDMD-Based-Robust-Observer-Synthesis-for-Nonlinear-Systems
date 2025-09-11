import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.patches as patches

# --- load matrices ---
A = np.load('A_matrix.npy')  # (3,3)
C = np.load('C_matrix.npy')  # (m,3)
L = np.load('L.npy')         # (3,m)
N = A.shape[0]

# --- scalar params for lifting map ---
rho = -2
lam = -1
den = lam - 2*rho
if np.isclose(den, 0.0):
    raise ValueError("lam - 2*rho == 0 -> lifting is singular.")
gamma = lam / den

def lift(x):
    x1, x2 = x
    return np.array([x1, x2, x2 - gamma*(x1**2)], dtype=float)

# --- coupled dynamics: true lifted plant + observer ---
def coupled_dynamics(t, s, A, C, L):
    z_true = s[0:N]
    z_hat  = s[N:2*N]

    dz_true = A @ z_true
    y       = C @ z_true
    dz_hat  = A @ z_hat + L @ (y - C @ z_hat)

    return np.hstack([dz_true, dz_hat])

# --- generate n random initial conditions inside the unit circle ---
def random_points_in_circle(n, radius=1.0, seed=None):
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(n))
    theta = rng.uniform(0, 2*np.pi, n)
    # r = radius * np.sqrt(np.random.uniform(0, 1, n))
    # theta = np.random.uniform(0, 2*np.pi, n)
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    return np.column_stack((x1, x2))

# --- simulation settings ---
t_span = [0, 5]
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# --- phase portrait with 10 ICs ---
fig1, ax = plt.subplots(figsize=(6, 6))
unit_circle = patches.Circle((0, 0), 1, facecolor="lightgray",
                             edgecolor="k", linestyle="--", linewidth=1.2, zorder=0)
ax.add_patch(unit_circle)
# --- plot for error norm decay ---
fig2, ax_err = plt.subplots(figsize=(7, 4))

initial_conditions = random_points_in_circle(10, radius=1.0, seed=22)
x1_true = np.zeros(t_eval.shape + (initial_conditions.shape[0],))
x2_true = np.zeros(t_eval.shape + (initial_conditions.shape[0],))
x1_hat  = np.zeros(t_eval.shape + (initial_conditions.shape[0],))
x2_hat  = np.zeros(t_eval.shape + (initial_conditions.shape[0],))
err_norm = np.zeros(t_eval.shape + (initial_conditions.shape[0],))
for idx, x0_true in enumerate(initial_conditions):
    # observer IC = perturbed true IC
    x0_hat = x0_true + 0.05*random_points_in_circle(1, radius=1.0, seed=idx).flatten()
    # lift to z
    z0_true = lift(x0_true)
    z0_hat  = lift(x0_hat)
    s0 = np.hstack([z0_true, z0_hat])
    # simulate
    sol = solve_ivp(lambda t, s: coupled_dynamics(t, s, A, C, L),
                    t_span, s0, t_eval=t_eval, method='RK45')
    # for error
    e_traj = sol.y[0:N,:] - sol.y[N:2*N,:]
    err_norm[:,idx], all_t = np.maximum(np.linalg.norm(e_traj, axis = 0), 1e-12), sol.t # use np.maximum to avoid log(0)
    ax_err.semilogy(sol.t, err_norm[:,idx], linewidth=1.0, alpha = 0.9, label=f'IC {idx+1}')
    # true trajectory
    x1_true[:,idx], x2_true[:,idx] = sol.y[0], sol.y[1]
    x1_hat[:,idx],  x2_hat[:,idx]  = sol.y[3], sol.y[4]
 
    line_true, = ax.plot(x1_true[:,idx], x2_true[:,idx], linewidth=1.0, zorder=2)
    c = line_true.get_color()
    ax.scatter(x1_true[0],  x2_true[0],  color=c, edgecolor='k', marker='o', s=30, zorder=3)
    ax.scatter(x1_true[-1], x2_true[-1], color=c, edgecolor='k', marker='x', s=40, zorder=3)
    # observer trajectory (dashed, same color)
    ax.plot(x1_hat, x2_hat, linestyle='--', color = c, linewidth=1.2, zorder=2)
 
np.save("err_norm_alpha2.npy", err_norm)
np.save("x1_true_alpha2.npy", x1_true)
np.save("x2_true_alpha2.npy", x2_true)
np.save("x1_hat_alpha2.npy", x1_hat)
np.save("x2_hat_alpha2.npy", x2_hat)
np.save("all_t.npy", all_t)
# labels, limits, grid
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_aspect('equal')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_xticks(np.arange(-1.2, 1.21, 0.2))
ax.set_yticks(np.arange(-1.2, 1.21, 0.2))
ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
fig1.tight_layout()
 

ax_err.set_yscale('log')
ax_err.set_xlabel(r'$t$')
ax_err.set_ylabel(r'$\|e(t)\|_2$')
ax_err.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
fig2.tight_layout()
plt.show()