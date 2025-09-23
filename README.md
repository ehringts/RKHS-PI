# RKHS-PI
Code for the paper "Recovery of the optimal control value function in reproducing kernel Hilbert spaces from verification conditions" by T. Ehring, B. Azmi and B. Haasdonk
# Organisation of the repository
In "\functions" the following files are stored:
- kernel.py: This class provides routines to compute the generalized Gramian matrix used in the RKHS-PI method, and evaluate the surrogate model and its gradient. Multiple kernel choices are supported.
- surrogate.py:  This class provides routines to generate a surrogate model for the value function in the RKHS-PI and fit the surrogate model to data in a greedy fashion.
- model.py: This class provides routines to implement optimal control problems of the form int_0^\infty h(x) + u^\top R u dt subject to x'=f(x)+g(x)u, x(0)=x0 and solve the corresponding open-loop boundary value problem.
- observer.py: This class provides routines to monitor the progress of the RKHS-PI algorithm and plot the results.
- auxFunctions.py: This functions implements the RKHS-PI algorithm and routines to find the best parameter gamma for the kernels used in the surrogate model. 
The files VanDerPol.py, ToyExample.py, LinearHEDirichlet1D.py, and NonlinearHEDirichlet1D.py provide scripts that generate a surrogate model for the value function of the corresponding model problem using the RKHS-PI method.
