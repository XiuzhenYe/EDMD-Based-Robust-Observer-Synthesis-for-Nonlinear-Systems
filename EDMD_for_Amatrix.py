import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import control

def system_dynamics(x, rho = -2, lam = -1):
    x1 = x[:,0]
    x2 = x[:,1]
    # System equations
    dx1_dt = rho * x1
    dx2_dt = lam * (x2 - x1**2)
    return np.stack([dx1_dt, dx2_dt], axis = 1)


d = 5000
dt = 0.001 # smaller dt yields better accuracy for A matrix
X = np.random.uniform(-1, 1, size=(d, 2))
 
X_plus = X + dt * system_dynamics(X)

# Koopman lifting, observables
def koopman_lift(X, lam = -1, rho = -2):
    x1 = X[:, 0]
    x2 = X[:, 1]
    return np.stack([x1, x2, x2 - lam/(lam - 2*rho)*(x1**2)], axis=1)

Phi = koopman_lift(X)
Phi_plus = koopman_lift(X_plus)

# least square estimation of A
Y = (Phi_plus - Phi) / dt 
 
A_T, _, _, _ = np.linalg.lstsq(Phi, Y, rcond=None)   
A = A_T.T
 
print(A) 


######## Define output matrix C and check observability ########
C = np.array([[1, 1, 0]])  
O = control.obsv(A, C)
is_obs = np.linalg.matrix_rank(O) == A.shape[0]
print("System observable? ", is_obs)

if is_obs:
    np.save('A_matrix.npy', A)
    np.save('C_matrix.npy', C)