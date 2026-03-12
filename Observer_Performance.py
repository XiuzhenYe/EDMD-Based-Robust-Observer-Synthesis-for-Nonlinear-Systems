import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.patches as patches

system_example_index = 2
if system_example_index == 1:
    initial_radius = 1.0
elif system_example_index == 2:
    initial_radius = 0.5
elif system_example_index == 3:
    initial_radius = 1.0
    poly_degree = 8

def polynomial_features(x, degree):
    y = []
    for d in range(1, degree+1):
        for i in range(d+1):
            y.append((x[0]**(d-i)) * (x[1]**i))
    return np.array(y)
def polynomial_features_grad(x, degree):
    grad_x1_list = []
    grad_x2_list = []
    for d in range(1, degree+1):
        for i in range(d+1):
            grad_x1 = (d-i) * (x[0]**(d-i-1)) * (x[1]**i) if d-i > 0 else 0
            grad_x2 = i * (x[0]**(d-i)) * (x[1]**(i-1)) if i > 0 else 0
            grad_x1_list.append(grad_x1)
            grad_x2_list.append(grad_x2)
    return np.array([grad_x1_list, grad_x2_list]).T

def state_dynamics(x):
    x1, x2 = x
    if system_example_index == 1: #Toy
        rho, lam = -2, -1
        dx1_dt = rho * x1
        dx2_dt = lam * (x2 - x1**2)
    elif system_example_index == 2: #Two tanks
        dx1_dt = 5*(2-x1) - 2.5*(2+x1)**2
        dx2_dt = -5*x2 + 5*(x1-x2) - (5/18)*((3+x2)**2 - 9)
    elif system_example_index == 3: #Van der Pol
        mu = 0.5
        dx1_dt = x2
        dx2_dt = mu * (1 - (x1*3)**2) * x2 - x1
    return np.array([dx1_dt, dx2_dt])

# --- load matrices ---
A, C, L = np.load('A_matrix.npy'), np.load('C_matrix.npy'), np.load('L.npy')  
N = A.shape[0]

# --- scalar params for lifting map ---
def koopman_lift(x):
    x1, x2 = x
    if system_example_index == 1:
        rho, lam = -2, -1 
        return np.array([x1, x2, x2 - lam/(lam - 2*rho)*(x1**2)], dtype=float)
    elif system_example_index == 2:
        return np.array([x1, x2, x1**2, x2**2, x1*x2], dtype=float)
    elif system_example_index == 3:
        return polynomial_features(x, degree=poly_degree) 

# --- coupled dynamics: true lifted plant + observer ---
def coupled_dynamics(t, s, A, C, L): 
    x, z_hat = s[0:2], s[2:] 
    dz_hat  = A @ z_hat + L @ C @ (koopman_lift(x) - z_hat)
    return np.hstack([state_dynamics(x), dz_hat])

# --- generate n random initial conditions inside the unit circle ---
def random_points_in_circle(n, radius=1.0, seed=None):
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(n))
    theta = rng.uniform(0, 2*np.pi, n) 
    x1, x2 = r * np.cos(theta), r * np.sin(theta) 
    return np.column_stack((x1, x2))

# --- simulation settings ---
t_eval = np.linspace(0, 10, 1000) 
fig1, ax = plt.subplots(figsize=(4, 4))
unit_circle = patches.Circle((0, 0), initial_radius, edgecolor="k", facecolor="None", linestyle="--", linewidth=1.2, zorder=0) #
ax.add_patch(unit_circle)
# --- plot for error norm decay ---
fig2, ax_err = plt.subplots(figsize=(4, 2))
initial_conditions = random_points_in_circle(n=10, radius=initial_radius, seed=42)
x1_true = np.zeros(t_eval.shape + (initial_conditions.shape[0],))
x2_true = np.zeros(t_eval.shape + (initial_conditions.shape[0],))
x1_hat  = np.zeros(t_eval.shape + (initial_conditions.shape[0],))
x2_hat  = np.zeros(t_eval.shape + (initial_conditions.shape[0],))
err_norm = np.zeros(t_eval.shape + (initial_conditions.shape[0],))
err_level = []

for idx, x0_true in enumerate(initial_conditions):
    # observer IC = perturbed true IC
    x0_hat = x0_true + 0.02*random_points_in_circle(1, radius=1.0, seed=idx).flatten()
    sol = solve_ivp(lambda t, s: coupled_dynamics(t, s, A, C, L), 
                    [t_eval[0], t_eval[-1]], np.hstack([x0_true, koopman_lift(x0_hat)]), 
                    t_eval=t_eval, method='RK45')
    print("Simulation for IC ", idx, " done.")
    x1_true[:,idx], x2_true[:,idx] = sol.y[0], sol.y[1]
    x1_hat[:,idx],  x2_hat[:,idx]  = sol.y[2], sol.y[3]
    
    line_true, = ax.plot(x1_true[:,idx], x2_true[:,idx], linewidth=1.0, zorder=2)
    c = line_true.get_color()
    ax.scatter(x1_true[0, idx],  x2_true[0, idx],  color=c, edgecolor='k', marker='o', s=30, zorder=3)
 
    # observer trajectory (dashed, same color)
    ax.plot(x1_hat[:,idx], x2_hat[:,idx], linestyle='--', color=c, linewidth=1.2, zorder=2)  # <-- fixed: [:,idx]
    
    # for error
    e_traj = sol.y[0:1,:] - sol.y[2:3,:]
    err_norm[:,idx], all_t = np.maximum(np.linalg.norm(e_traj, axis=0), 1e-12), sol.t
    ax_err.semilogy(sol.t, err_norm[:,idx], color=c, linewidth=1.0, alpha=0.9)  # <-- also added color=c here
    err_level.append(np.mean(err_norm[:, idx]**2))

print("Average error level across all trajectories (RMSE): ", np.sqrt(np.mean(err_level)))
np.save("err_norm_alpha2.npy", err_norm)
np.save("x1_true_alpha2.npy", x1_true)
np.save("x2_true_alpha2.npy", x2_true)
np.save("x1_hat_alpha2.npy", x1_hat)
np.save("x2_hat_alpha2.npy", x2_hat)
np.save("all_t.npy", all_t) 

ax.set_xlabel('$x_1$'), ax.set_ylabel('$x_2$'), ax.set_aspect('equal')
ax.set_xlim(-1.2*initial_radius, 1.2*initial_radius), ax.set_ylim(-1.2*initial_radius, 1.2*initial_radius)
fig1.tight_layout()

ax_err.set_yscale('log')
ax_err.set_xlabel(r'$t$'), ax_err.set_ylabel(r'$\|e(t)\|_2$')
# ax_err.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
fig2.tight_layout()
plt.show()