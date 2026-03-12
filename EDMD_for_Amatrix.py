# from platform import system
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import control

system_example_index = 2
if system_example_index == 1:
    initial_radius = 1.0
elif system_example_index == 2:
    initial_radius = 0.5
elif system_example_index == 3:
    initial_radius = 1.0
    poly_degree = 8

def polynomial_features(X, degree):
    X_poly = []
    for d in range(1, degree+1):
        for i in range(d+1):
            X_poly.append((X[:, 0]**(d-i)) * (X[:, 1]**i))
    return np.stack(X_poly, axis=1)
def polynomial_features_grad(X, degree):
    grad_x1_list = []
    grad_x2_list = []
    for d in range(1, degree+1):
        for i in range(d+1):
            grad_x1 = (d-i) * (X[:, 0]**(d-i-1)) * (X[:, 1]**i) if d-i > 0 else np.zeros_like(X[:, 0])
            grad_x2 = i * (X[:, 0]**(d-i)) * (X[:, 1]**(i-1)) if i > 0 else np.zeros_like(X[:, 1])
            grad_x1_list.append(grad_x1)
            grad_x2_list.append(grad_x2)
    return np.stack(grad_x1_list, axis=1), np.stack(grad_x2_list, axis=1)

def system_dynamics(x):
    x1 = x[:,0]
    x2 = x[:,1]
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
    return np.stack([dx1_dt, dx2_dt], axis = 1)

def koopman_lift(X, lam = -1, rho = -2):
    x1 = X[:, 0]
    x2 = X[:, 1]
    if system_example_index == 1: #Toy
        # return np.stack([x1, x2, x2 - lam/(lam - 2*rho)*(x1**2)], axis=1)
        return np.stack([x1, x2, x2 - lam/(lam - 2*rho)*(x1**2)], axis=1)  
    elif system_example_index == 2: #Two tanks
        return np.stack([x1, x2, x1**2, x2**2, x1*x2], axis=1)
    elif system_example_index == 3: #Van der Pol
        return polynomial_features(X, degree=poly_degree)
def koopman_lift_grad(X, lam = -1, rho = -2):
    x1 = X[:, 0]
    x2 = X[:, 1]
    if system_example_index == 1: #Toy
        dphi_dx1 = np.stack([np.ones_like(x1), np.zeros_like(x1), -2*lam/(lam - 2*rho)*x1], axis=1)
        dphi_dx2 = np.stack([np.zeros_like(x2), np.ones_like(x2), np.ones_like(x2)], axis=1)
        return dphi_dx1, dphi_dx2
    elif system_example_index == 2: #Two tanks
        dphi_dx1 = np.stack([np.ones_like(x1), np.zeros_like(x1), 2*x1, np.zeros_like(x1), x2], axis=1)
        dphi_dx2 = np.stack([np.zeros_like(x2), np.ones_like(x2), np.zeros_like(x2), 2*x2, x1], axis=1)
        return dphi_dx1, dphi_dx2
    elif system_example_index == 3: #Van der Pol
        return polynomial_features_grad(X, degree=poly_degree)

def try_sampling_size(M=500, initial_radius=1.0, seed=None):
    X = np.random.uniform(-initial_radius, initial_radius, size=(M, 2))
    X_dot = system_dynamics(X) # real dynamics
    Phi = koopman_lift(X) # lift via functions in the chosen dictionary
    dphi_dx1, dphi_dx2 = koopman_lift_grad(X)
    Phi_dot = dphi_dx1 * X_dot[:, 0:1] + dphi_dx2 * X_dot[:, 1:2] # chain rule
    A_T, _, _, _ = np.linalg.lstsq(Phi, Phi_dot, rcond=None)   
    A = A_T.T
    return A, Phi, Phi_dot

####### Try a range of sampling size
M_range = [int(x) for x in np.logspace(1, 4, 31)]
num_experiments = 50
trace_records = np.zeros((len(M_range), num_experiments))
for count in range(len(M_range)):
    M = M_range[count]
    for exper in range(num_experiments):
        A, _, _ = try_sampling_size(M=M, initial_radius=initial_radius, seed=exper)
        trace_records[count, exper] = np.trace(A)
plt.figure(figsize=(5, 5))
plt.scatter(np.repeat(M_range, num_experiments), trace_records.flatten(), alpha=0.6)
plt.xscale('log'), plt.xlabel(r'$M$'), plt.ylabel(r'$\mathrm{trace}(\mathcal{L}_{N,M})$')
plt.show() 

A, Phi, Phi_dot = try_sampling_size(M=2000, initial_radius=initial_radius, seed=42) 

######## Estimate the relative error 
phi_sample = np.sum(Phi**2, axis=1)**0.5
r_sample = np.sum((Phi_dot - Phi @ A.T)**2, axis=1)**0.5
c_r_estim = np.max(r_sample/phi_sample)
print("Estimation for c_r = ", c_r_estim)
plt.figure(figsize=(5, 4))
plt.scatter(phi_sample, r_sample, alpha=0.6, s=12)
plt.plot([0, np.max(phi_sample)], [0, c_r_estim*np.max(phi_sample)], 'r--') 
plt.plot([0, np.max(phi_sample)], [0, 0], 'k--') 
plt.xlabel(r'$\|\phi(x_i)\|$'), plt.ylabel(r'$\|r(x_i)\|$'), plt.show()

######## Define output matrix C and check observability ########
if system_example_index == 1: #Toy
    C = np.array([[1, 1, 0]])  
elif system_example_index == 2: #Two tanks
    C = np.array([[0, 1, 1, 0, 0]]) 
elif system_example_index == 3: #Van der Pol
    C = np.zeros((1, Phi.shape[1]))
    C[0, 1] = 1.0
    # C = []
    # start, incr = 1, 3
    # while start < Phi.shape[1]:
    #     e = np.zeros(Phi.shape[1])
    #     e[start] = 1.0
    #     start += incr
    #     incr += 1
    #     C.append(e)
    # C = np.array(C)
O = control.obsv(A, C) 

print("Eigenvalues of A: ", np.linalg.eigvals(A))
print("System observable? ", np.linalg.matrix_rank(O) == A.shape[0])

np.save('A_matrix.npy', A)
np.save('C_matrix.npy', C) 