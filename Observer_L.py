import numpy as np
import cvxpy as cp
import scipy.linalg as la

system_example_index = 2
# Desired convergence rate
alpha = 1
cr = 0.67

# --- data ---
A = np.load('A_matrix.npy')
C = np.load('C_matrix.npy')
m = C.shape[0]  # number of outputs
N = A.shape[0]
I = np.eye(N)
sym = lambda X: (X + X.T) / 2
eps_P, eps_M = 1e-1, 1e-4
# --- variables (no bilinearities with alpha anymore) ---
Pphi = cp.Variable((N, N), PSD=True)
Pe   = cp.Variable((N, N), PSD=True)
G    = cp.Variable((N, m)) # G = Pe * L
lam  = cp.Variable(nonneg=True)
nu = cp.Variable(nonneg=True)
# initialized values
I2 = np.zeros((N, N))
I2[0, 0] = 1.0
I2[1, 1] = 1.0

# --- build problem ---
def build_problem(alpha_param):
    Z = cp.Constant(np.zeros((N, N))) 
    TL = sym(Pphi @ A + A.T @ Pphi + 2*alpha_param*Pphi + lam*(cr**2)*I)
    MM = sym(Pe @ A + A.T @ Pe - G @ C - C.T @ G.T + 2*alpha_param*Pe)
    M = sym(cp.bmat([[TL, Z, Pphi], [Z, MM, Pe], [Pphi, Pe, -lam*I]]))
    cons = [
        Pphi >> eps_P*I,
        Pe   >> eps_P*I,
        M    << -eps_M*np.eye(3*N),
        lam  >= 1e-3,
        cp.trace(Pphi) + cp.trace(Pe) == 1
    ]
    return cp.Problem(cp.Minimize(lam), cons)
def build_problem_L2(): 
    # The following naive formulation does not work well. The problem is likely ill-conditioned. 
    cons = [
        Pe >> 0*I, 
        cp.bmat([[Pe @ A + A.T @ Pe - G @ C - C.T @ G.T + nu*I2, Pe], [Pe, -lam*I]]) + 0*np.eye(2*N) << 0,
        nu >= 1e-3
    ]     
    return cp.Problem(cp.Minimize(lam), cons)

# --- try solving for a given alpha ---
def try_solve(alpha_val):
    if system_example_index == 3:
        prob = build_problem_L2()
    else:
        prob = build_problem(alpha_val)
    prob.solve(solver = cp.SCS, verbose=True, eps=1e-4, max_iters=100000, warm_start=True)
    feas = prob.status in ("optimal", "optimal_inaccurate")
    status = prob.status
    return feas, status

# --- main: try a specific alpha ---
feas, status = try_solve(alpha)
if status == "optimal":
    P_value = Pe.value
    L_val = (np.linalg.pinv(P_value + 1e-4*I))@ G.value 
    if system_example_index == 3:
        print("eigenvalue of Pe:", la.eigvals(sym(Pe.value))) 
        print("Optimal L2 gain = ", np.sqrt(lam.value/nu.value))
if status in ("optimal_inaccurate", "infeasible"):
    print("Warning: solver returned optimal_inaccurate. Use CARE instead...")
    # Resort to Kalman filter by ARE
    R_tuning = 1
    X_riccati = la.solve_continuous_are(A.T, C.T, I2, R_tuning)
    L_val = (1/R_tuning) * X_riccati @ C.T
print("Eigenvalues of A-LC:\n", la.eigvals(A - L_val @ C))
np.save("L.npy", L_val)