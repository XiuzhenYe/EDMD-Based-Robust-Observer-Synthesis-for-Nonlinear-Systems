import numpy as np
import cvxpy as cp
import scipy.linalg as la

# --- data ---
A = np.load('A_matrix.npy')
C = np.load('C_matrix.npy')
m = C.shape[0]  # number of outputs
N = A.shape[0]
I = np.eye(N)
cr = 0.1
def sym(X): return (X + X.T) / 2
eps_P = 1e-5
eps_M = 1e-10
# --- variables (no bilinearities with alpha anymore) ---
Pphi = cp.Variable((N, N), PSD=True)
Pe   = cp.Variable((N, N), PSD=True)
G    = cp.Variable((N, m))        # G = Pe * L
lam  = cp.Variable(nonneg=True)
# --- build problem ---
def build_problem(alpha_param):
    Z = cp.Constant(np.zeros((N, N))) 
    TL = sym(Pphi @ A + A.T @ Pphi + 2*alpha_param*Pphi + lam*(cr**2)*I)
    TM = Z
    TR = Pphi
    ML = Z
    MM = sym(Pe @ A + A.T @ Pe - G @ C - C.T @ G.T + 2*alpha_param*Pe)
    MR = Pe
    BL = Pphi
    BM = Pe
    BR = -lam*I
    M = sym(cp.bmat([
        [TL, TM, TR],
        [ML, MM, MR],
        [BL, BM, BR]
    ]))
    cons = [
        Pphi >> eps_P*I,
        Pe   >> eps_P*I,
        M    << -eps_M*np.eye(3*N),
        lam  >= 1e-3,
        cp.trace(Pphi) + cp.trace(Pe) == 1
    ]
    return cp.Problem(cp.Minimize(lam), cons)
# --- try solving for a given alpha ---
def try_solve(alpha_val, solver="MOSEK", verbose=True):
    prob = build_problem(alpha_val)
    prob.solve(solver = cp.SCS, max_iters = 200000, verbose=False)
    feas = prob.status in ("optimal", "optimal_inaccurate")
    status = prob.status
    return feas, status
# --- main: try a specific alpha ---
alpha = 1
feas, status = try_solve(alpha, solver="MOSEK", verbose=False)
print(status)
if feas:
    Pphi_val = sym(Pphi.value)
    Pe_val   = sym(Pe.value)
    G_val    = G.value
    lam_val  = lam.value
    L_val = np.linalg.pinv(Pe_val) @ G_val  # alternative, more numerically stable
    # --- reconstruct numeric M ---
    TL_val = Pphi_val @ A + A.T @ Pphi_val \
             + 2 * alpha * Pphi_val \
             + lam_val * (cr**2) * np.eye(N)
    TM_val = np.zeros((N, N))
    TR_val = Pphi_val
    ML_val = np.zeros((N, N))
    MM_val = Pe_val @ A + A.T @ Pe_val - G_val @ C - C.T @ G_val.T \
             + 2 * alpha * Pe_val 
    MR_val = Pe_val
    BL_val = Pphi_val
    BM_val = Pe_val
    BR_val = -lam_val * np.eye(N)

    M_val = sym(np.block([
        [TL_val, TM_val, TR_val],
        [ML_val, MM_val, MR_val],
        [BL_val, BM_val, BR_val]
    ]))

    print("lambda:", lam_val)
    print("Pphi:\n", Pphi_val)
    print("eigenvalue of Pphi:", la.eigvals(Pphi_val))
    print("Pe:\n", Pe_val)
    print("eigenvalue of Pe:", la.eigvals(Pe_val))
    print("L (observer gain):\n", L_val)
    print("Eigenvalues of A - L C:\n", la.eigvals(A - L_val @ C))
np.save("L.npy", L_val)
 
