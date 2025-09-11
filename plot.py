import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

# --- load data ---
err_norm_alpha1 = np.load('err_norm_alpha1.npy')
x1_hat_alpha1 = np.load('x1_hat_alpha1.npy')
x2_hat_alpha1 = np.load('x2_hat_alpha1.npy')
x1_true_alpha1 = np.load('x1_true_alpha1.npy')
x2_true_alpha1 = np.load('x2_true_alpha1.npy')

err_norm_alpha2 = np.load('err_norm_alpha2.npy')
x1_hat_alpha2 = np.load('x1_hat_alpha2.npy')
x2_hat_alpha2 = np.load('x2_hat_alpha2.npy')
x1_true_alpha2 = np.load('x1_true_alpha2.npy')
x2_true_alpha2 = np.load('x2_true_alpha2.npy')

all_t = np.load('all_t.npy')
n = err_norm_alpha1.shape[1]
colors = cm.get_cmap("tab10", n)

# --- helper for phase portrait ---
def plot_phase(ax, x1_true, x2_true, x1_hat, x2_hat, alpha_label):
    unit_circle = patches.Circle((0, 0), 1, facecolor="lightgray",
                                 edgecolor="k", linestyle="--", linewidth=1.2, zorder=0)
    ax.add_patch(unit_circle)

    for idx in range(n):
        color = colors(idx)
        ax.plot(x1_true[:, idx], x2_true[:, idx], linewidth=1.0, zorder=2, color=color)
        ax.scatter(x1_true[0, idx],  x2_true[0, idx],  color=color,
                   edgecolor='k', marker='o', s=30, zorder=3)
        ax.scatter(x1_true[-1, idx], x2_true[-1, idx], color=color,
                   edgecolor='k', marker='x', s=40, zorder=3)
        ax.plot(x1_hat[:, idx], x2_hat[:, idx], linestyle='--',
                color=color, linewidth=1.2, zorder=2)

    ax.set_title(rf'$\alpha = {alpha_label}$')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xticks(np.arange(-1.2, 1.21, 0.2))
    ax.set_yticks(np.arange(-1.2, 1.21, 0.2))
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

# --- helper for error norm ---
def plot_error(ax, err_norm, alpha_label):
    for idx in range(n):
        color = colors(idx)
        ax.semilogy(all_t, err_norm[:, idx], linewidth=1.0, alpha=0.9,
                    color=color, label=f'IC {idx+1}')
    ax.set_title(rf'$\alpha = {alpha_label}$')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\|e(t)\|_2$')
    ax.set_xlim(-0.2, all_t[-1]+0.2)
    ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)

# --- Figure 1: Phase portraits ---
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
plot_phase(ax1, x1_true_alpha1, x2_true_alpha1, x1_hat_alpha1, x2_hat_alpha1, "0.1")
plot_phase(ax2, x1_true_alpha2, x2_true_alpha2, x1_hat_alpha2, x2_hat_alpha2, "1")
fig1.tight_layout()

# --- Figure 2: Error norms ---
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))  # no sharey
plot_error(ax3, err_norm_alpha1, "0.1")
plot_error(ax4, err_norm_alpha2, "1")
fig2.tight_layout()

plt.show()
