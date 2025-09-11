import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.patches as patches

def system_dynamics(t, x, rho, lam, u):
    x1, x2 = x
    dx1_dt = rho * x1
    dx2_dt = lam * (x2 - x1**2) + (u(t) if callable(u) else u)
    return [dx1_dt, dx2_dt]

# Parameters
rho = -2
lam = -1

# Simulation parameters
t_span = [0, 5]
t_eval = np.linspace(t_span[0], t_span[1], 1000)
n = 10
# Control input
def zero_input(t):
    return 0

# Initial conditions
def random_points_in_circle(n, radius=1.0, seed=None):
    rng = np.random.default_rng(seed)
    # Random radius (sqrt for uniform distribution inside circle)
    r = radius * np.sqrt(rng.random(n))
    # Random angle
    theta = rng.uniform(0, 2*np.pi, n)
    # Convert to Cartesian coordinates
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    return np.column_stack((x1, x2))

# Phase portrait only
fig, ax = plt.subplots(figsize=(6, 6))

# Draw filled unit circle (gray interior, dashed black edge)
unit_circle = patches.Circle(
    (0, 0), 1, facecolor="lightgray",
    edgecolor="k", linestyle="--", linewidth=1.2, zorder=0
)
ax.add_patch(unit_circle)

for x0 in random_points_in_circle(n,radius = 1.0, seed=42):
    sol = solve_ivp(
        lambda t, x: system_dynamics(t, x, rho, lam, zero_input),
        t_span, x0, t_eval=t_eval, method='RK45'
    )

    line, = ax.plot(sol.y[0], sol.y[1], label=f'IC=({x0[0]:.2f},{x0[1]:.2f})', zorder=2)
    c = line.get_color()
    ax.scatter(sol.y[0, 0], sol.y[1, 0], color=c, edgecolor='k', marker='o', s=60, zorder=3)  # start
    ax.scatter(sol.y[0, -1], sol.y[1, -1], color=c, edgecolor='k', marker='x', s=60, zorder=3)  # end

# Labels and grid
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_aspect('equal')
# finer grid (every 0.2)
ax.set_xticks(np.arange(-1.2, 1.21, 0.15))
ax.set_yticks(np.arange(-1.2, 1.21, 0.15))
ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)  # grid on
ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.2, 1.3)
 

plt.tight_layout()
plt.show()
