# EDMD-Based Robust Observer Synthesis for Nonlinear Systems

This repository contains the implementation for the paper:

> **"EDMD-Based Robust Observer Synthesis for Nonlinear Systems"**
> Xiuzhen Ye, Wentao Tang — submitted to CDC 2026

## Overview
This project provides a data-driven framework for designing robust state observers 
for nonlinear systems using the Extended Dynamic Mode Decomposition (EDMD) and 
Koopman operator theory.

## Steps to Design Your Robust Observer

### Step 1: Identify the Koopman Matrix A — `EDMD_for_Amatrix.py`
1. Specify your system dynamics in `system_dynamics`.
2. Choose a Koopman lifting dictionary of N observable functions.
3. Set a range of sampling sizes M to study convergence.
4. Run the script to obtain:
   - The approximated Koopman matrix **A** (i.e., $\mathcal{L}_{N,M}$)
   - The estimated sectorial error bound **c_r**

### Step 2: Compute the Observer Gain L — `Observer_L.py`
1. Set the desired exponential convergence rate `alpha > 0`.
   - Larger `alpha` → more aggressive observer convergence.
2. Set `cr` from Step 1.
3. Run the script to obtain the observer gain **L**.
   - Verify observer stability from the printed eigenvalues of **A - LC**.
   - The observer error converges to 0 if all eigenvalues of **A - LC** have negative real parts.

### Step 3: Evaluate Observer Performance — `Observer_Performance.py`
1. Load matrices **A**, **C**, and observer gain **L**.
2. Specify the system dynamics consistent with Step 1.
3. Run the script to visualize:
   - True state trajectories vs. observer estimates
   - Observer error decay over time

## Requirements
- Python 3.x
- `numpy`, `scipy`, `cvxpy`, `matplotlib`, `control`

## Citation
If you find this code useful, please cite:
> Ye, X., Tang, W., "EDMD-Based Robust Observer Synthesis for Nonlinear Systems", CDC 2026.
