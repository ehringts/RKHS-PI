import numpy as np
from   functions import kernel, kernelCon, model,surrogate, observer, auxFunctions
import matplotlib.pyplot    as plt
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import numpy as np
from   functions import kernel, model,surrogate, observer, auxFunctions
import matplotlib.pyplot    as plt
import numpy as np
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve
from scipy.optimize import minimize, Bounds
import pickle
import time


"""
This script compares two numerical approaches for approximating a value function
on a two-dimensional grid.

The first approach solves a constrained optimization problem using SLSQP. It
minimizes an RKHS norm subject to pointwise nonlinear equality constraints and
a positivity constraint on the value function.

The second approach uses an unconstrained RKHS-based policy iteration method.
It repeatedly updates the control law from the current surrogate approximation
and refits the value function on the grid.

For different values of the parameter zeta, both methods are evaluated against
two reference solutions: a true solution and a spurious solution. The script
computes the relative errors, classifies which reference solution each method
converges to, and visualizes the results in a convergence plot.
"""


def ConRun(a):
    x = np.linspace(-0.5, 0.5, 10)
    X, Y = np.meshgrid(x, x)
    centers = np.vstack([X.ravel(), Y.ravel()]).astype(np.float64)

    newkernel = kernelCon.LinMaternProduct(0.25) #1.1
    S = centers
    N, M = S.shape

    Y = np.tile(S, N + 1)                              # shape (N, M*(N+1))
    funcYList = np.repeat(np.arange(N + 1, dtype=int), M)

    K = newkernel.getGramHermite(Y, funcYList, Y, funcYList)
    print(np.linalg.cond(K))
    # ------------------------------------------------------------
    # factor K once
    # ------------------------------------------------------------
    try:
        Kfac = cho_factor(K, lower=True, check_finite=False)
        def Ksolve(b):
            return cho_solve(Kfac, b, check_finite=False)
    except np.linalg.LinAlgError:
        Kfac = lu_factor(K, check_finite=False)
        def Ksolve(b):
            return lu_solve(Kfac, b, check_finite=False)

    # Precompute things used by the constraint
    

    idx = np.arange(M)

    r2  = np.einsum("ij,ij->j", centers, centers)   # shape (M,)
    er  = np.exp(r2)
    e2r = np.exp(2.0 * r2)
    idx = np.arange(M)

    def unpack_x(z):
        return z[M:].reshape(N, M)   # here N should be 2

    # ------------------------------------------------------------
    # objective + gradient together
    # unchanged
    # ------------------------------------------------------------
    def fun_and_jac(z):
        y = Ksolve(z)                  # y = K^{-1} z
        f = float(z @ y)               # z^T K^{-1} z
        g = 2.0 * y
        return f, g

    # ------------------------------------------------------------
    # equality constraint for the new problem (beta = 1)
    # ------------------------------------------------------------
    def eqCon(z):
        x = unpack_x(z)                # x[0,:] = p1, x[1,:] = p2

        c1 = centers[0, :]
        c2 = centers[1, :]

        return (
            0.5 * e2r * c1**2
            + (0.25 * er * c1 - c2) * x[0, :]
            + c1 * x[1, :]
            - 0.25 * x[0, :]**2
        )

    def eqCon_jac(z):
        """
        Jacobian shape: (M, M + N*M)

        For each center j, the constraint is
        c_j = 0.5*exp(2*r2_j)*c1_j^2
                + (0.25*exp(r2_j)*c1_j - c2_j)*x1_j
                + c1_j*x2_j
                - 0.25*x1_j^2

        Hence
        d c_j / d x1_j = 0.25*exp(r2_j)*c1_j - c2_j - 0.5*x1_j
        d c_j / d x2_j = c1_j
        """
        x = unpack_x(z)

        c1 = centers[0, :]
        c2 = centers[1, :]

        J = np.zeros((M, M + N * M), dtype=np.float64)

        G0 = 0.25 * er * c1 - c2 - 0.5 * x[0, :]   # d/dx1
        G1 = c1                                    # d/dx2

        # row-major flattening of z[M:]:
        # x[0,:], then x[1,:]
        J[idx, M + 0 * M + idx] = G0
        J[idx, M + 1 * M + idx] = G1

        return J
    # ------------------------------------------------------------
    # initial point
    # ------------------------------------------------------------
    P0 = np.diag([a, a])

    z0 = np.r_[
        np.einsum("ij,ij->j", centers, P0 @ centers),
        (2.0 * P0 @ centers).ravel()
    ]

    # ------------------------------------------------------------
    # bounds
    # ------------------------------------------------------------
    ineqOn = True
    if ineqOn:
        lb = np.r_[np.zeros(M), -np.inf * np.ones(N * M)]
    else:
        lb = -np.inf * np.ones(M + N * M)

    ub = np.inf * np.ones(M + N * M)
    bounds = Bounds(lb, ub)

    # SLSQP prefers old-style dict constraints with explicit jac
    cons = [{"type": "eq", "fun": eqCon, "jac": eqCon_jac}]

    # ------------------------------------------------------------
    # solve
    # ------------------------------------------------------------
    res = minimize(
        fun_and_jac,
        z0,
        method="SLSQP",
        jac=True,          # tells scipy that fun returns (f, g)
        bounds=bounds,
        constraints=cons,
        options={
            "disp": True,
            "maxiter": 10000,
            "ftol": 1e-12,
        },
    )

    print("success:", res.success)
    print("message:", res.message)
    print("fun:", res.fun)
    print("||eq||_inf:", np.max(np.abs(eqCon(res.x))))
    print("min z[:M]:", np.min(res.x[:M]))

    A1 = np.exp(np.sum(centers * centers, 0)) - 1
    A2 = 0.5 * (1 - np.exp(np.sum(centers * centers, 0)))

    print( np.sum(np.abs(res.x[:M] - A1))/centers.shape[1] )
    print( np.sum(np.abs(res.x[:M] - A2))/centers.shape[1] )


    newModel     = model.ConUnconExample()
    X,Y          = np.meshgrid(x,x)
    testPoint    = np.array([X.flatten(),Y.flatten()],dtype=np.float64)

    VFinput      = testPoint
    VFoutputTrue = newModel.trueVF(VFinput)
    VFoutputFalse = newModel.falseVF(VFinput)

    return np.sqrt( np.sum(np.abs(VFoutputTrue-res.x[:M])**2)/np.sum(np.abs(VFoutputTrue)**2)),np.sqrt( np.sum(np.abs(VFoutputFalse-res.x[:M])**2)/np.sum(np.abs(VFoutputFalse)**2)), res.x[:M]

def UnconRun(a):
    nMaxPI       = 10
    newModel     = model.ConUnconExample(a)
    x            = np.linspace(-0.5,0.5,10)
    X,Y          = np.meshgrid(x,x)
    testPoint    = np.array([X.flatten(),Y.flatten()],dtype=np.float64)

    VFinput       = testPoint
    VFoutputTrue  = newModel.trueVF(VFinput)
    VFoutputFalse = newModel.falseVF(VFinput)
    gamma      = 1/4
    newKernel  = kernel.LinMaternProduct(gamma,2)
    newSr      = surrogate.SurrogateProductKernel(newKernel)


    for j in range(nMaxPI):
        if  j==0:
            mu              = lambda x: newModel.stableControl(x)
            F               = lambda x: newModel.getF(x,mu(x))
            rhs             = lambda x: newModel.getRHS(x,mu(x))
            newSr.fit(F,rhs,testPoint, iterativ= False)
            maxFGreedyError = 0
            oldSRVal        = 0
        else:
            mu              = lambda x: newModel.getMuFromSr(x,newSr.kernel.evalGrad(FCenterVal,x,center,alpha))
            F               = lambda x: newModel.getF(x,mu(x))
            rhs             = lambda x: newModel.getRHS(x,mu(x)) 
            newSr.fit(F,rhs,testPoint, iterativ= False)
            maxFGreedyError = np.max(np.abs(newSr.kernel.getGramPDE(F,testPoint,newSr.center)@newSr.alpha-rhs(testPoint))/(np.abs(rhs(testPoint))+10**(-8)))
        FCenterVal = F(testPoint).copy()
        center     = testPoint.copy()
        alpha      = newSr.alpha.copy()
        trueError  = np.sqrt( np.sum(np.abs(VFoutputTrue-newSr.kernel.evalFunc(FCenterVal,VFinput,center,alpha))**2)/np.sum(np.abs(VFoutputTrue)**2))
        falseError = np.sqrt( np.sum(np.abs(VFoutputFalse-newSr.kernel.evalFunc(FCenterVal,VFinput,center,alpha))**2)/np.sum(np.abs(VFoutputFalse)**2))

        evalSR     = newSr.kernel.evalFunc(FCenterVal,testPoint,center,alpha)  
        isPos      = np.min(evalSR)>=0

        print(str(j) + " Iteration: True-Error = " + str( trueError ) + ", Residual-Error = " + str(maxFGreedyError) +", stagnation-Error = " + str( np.max(np.abs(oldSRVal-newSr.kernel.evalFunc(FCenterVal,testPoint,center,alpha))) )+", is positive = " + str(isPos))
        print(str(j) + " Iteration: False-Error = " + str( falseError ) + ", Residual-Error = " + str(maxFGreedyError) +", stagnation-Error = " + str( np.max(np.abs(oldSRVal-newSr.kernel.evalFunc(FCenterVal,testPoint,center,alpha))) )+", is positive = " + str(isPos))

        oldSRVal = newSr.kernel.evalFunc(FCenterVal,testPoint,center,alpha)    
    return trueError, falseError, newSr.kernel.evalFunc(FCenterVal,testPoint,center,alpha)    


def classify(err_true, err_false, tol=1e-8):
    if err_true < err_false - tol:
        return 1
    elif err_false < err_true - tol:
        return -1
    else:
        return 0

# --------------------------------------------------
# Sweep over parameter a
# --------------------------------------------------
def run_sweep(a_values):
    con_true = []
    con_false = []
    con_choice = []
    con_sol = []

    unc_true = []
    unc_false = []
    unc_choice = []
    unc_sol = []

    for k, a in enumerate(a_values):
        print(f"[{k+1}/{len(a_values)}] a = {a:.4f}")

        # constrained
        eT_c, eF_c, sol_c = ConRun(a)
        con_true.append(eT_c)
        con_false.append(eF_c)
        con_choice.append(classify(eT_c, eF_c))
        con_sol.append(sol_c.copy())

        # unconstrained
        eT_u, eF_u, sol_u = UnconRun(a)
        unc_true.append(eT_u)
        unc_false.append(eF_u)
        unc_choice.append(classify(eT_u, eF_u))
        unc_sol.append(sol_u.copy())

    return {
        "a": np.array(a_values),

        "con_true": np.array(con_true),
        "con_false": np.array(con_false),
        "con_choice": np.array(con_choice),
        "con_sol": con_sol,

        "unc_true": np.array(unc_true),
        "unc_false": np.array(unc_false),
        "unc_choice": np.array(unc_choice),
        "unc_sol": unc_sol,
    }

# --------------------------------------------------
# Main figure
# --------------------------------------------------
def plot_convergence_vs_parameter(
    results,
    title="Convergence versus parameter",
    filename="convergence_vs_parameter",
    save=True,
    show=True,
    mark_switch=True
):
    rc = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": [
            "Computer Modern Roman",
            "CMU Serif",
            "Latin Modern Roman",
            "DejaVu Serif",
        ],
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 1.6,
        "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
    }

    with plt.rc_context(rc):
        wid = 1.6
        ms = 3.0   # small marker size

        color_std = '#1f77b4'
        color_quad = '#ff7f0e'

        a = np.asarray(results["a"], dtype=float)
        con_true  = np.asarray(results["con_true"], dtype=float)
        con_false = np.asarray(results["con_false"], dtype=float)
        unc_true  = np.asarray(results["unc_true"], dtype=float)
        unc_false = np.asarray(results["unc_false"], dtype=float)

        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        fig.subplots_adjust(
            left=0.13,
            right=0.98,
            bottom=0.16,
            top=0.92
        )

        # -------------------------------------------------
        # Unconstrained
        # -------------------------------------------------
        ax.semilogy(
            a, unc_true,
            color=color_std,
            linestyle='-',
            linewidth=wid,
            #marker='o',
            markersize=ms,
            label=r'$\mathrm{TrueError}$ for RKHS-PI'
        )


        # -------------------------------------------------
        # Constrained
        # -------------------------------------------------
        ax.semilogy(
            a, con_true,
            color=color_quad,
            linestyle='-',
            linewidth=wid,
            #marker='o',
            markersize=ms,
            label=r'$\mathrm{TrueError}$ for SLSQP'
        )

        ax.semilogy(
            a, unc_false,
            color=color_std,
            linestyle='--',
            linewidth=wid,
            marker='o',
            markersize=ms,
            label=r'$\mathrm{SpuriousError}$ for RKHS-PI'
        )
                
        ax.semilogy(
            a, con_false,
            color=color_quad,
            linestyle='--',
            linewidth=wid,
            marker='o',
            markersize=ms,
            label=r'$\mathrm{SpuriousError}$ for SLSQP'
        )



        # -------------------------------------------------
        # Approximate switching point
        # -------------------------------------------------
        if mark_switch:
            unc_is_true = unc_true < unc_false
            switch_idx = np.where(np.diff(unc_is_true.astype(int)) != 0)[0]

            if len(switch_idx) > 0:
                i = switch_idx[0]
                a_switch = 0.5 * (a[i] + a[i + 1])

                ax.axvline(
                    a_switch,
                    color='black',
                    linestyle=':',
                    linewidth=1.4
                )

                y_min = np.nanmin(np.r_[con_true, con_false, unc_true, unc_false])
                y_max = np.nanmax(np.r_[con_true, con_false, unc_true, unc_false])

                y_text = np.exp(np.log(y_min) + 0.82 * (np.log(y_max) - np.log(y_min)))

                ax.text(
                    a_switch + 0.015,
                    y_text,
                    rf'$\zeta \approx {a_switch:.3f}$',
                    ha='left',
                    va='center'
                )

                print(f"Approximate switching point: \zeta ≈ {a_switch:.4f}")

        ax.set_title(title)
        ax.set_xlabel(r'parameter $\zeta$')
        ax.set_ylabel(r'relative error')
        ax.grid(True)
        ax.legend(loc='best', frameon=True)

        if save:
            plt.savefig(
                f"{filename}.pdf",
                bbox_inches='tight',
                pad_inches=0.02,
                dpi=1200
            )

        if show:
            plt.show()
        else:
            plt.close(fig)

# --------------------------------------------------
# Run everything
# --------------------------------------------------

a_values = np.r_[np.linspace(0.025, 0.25, 10),np.linspace(0.26, 0.3, 5),np.linspace(0.325, 0.5, 8)]   

# time_start = time.time()
a0,b0,c0 = ConRun(0.5)
a1,b1,c1 = UnconRun(0.5)

print(c0-c1)


# print(f"Time taken: {time.time() - time_start:.2f} seconds")

# print(a_values)

# results = run_sweep(a_values)

# with open("data/ConUnconRun", 'wb') as outp:
#     pickle.dump((a_values,results), outp, protocol=4)


with open("data/ConUnconRun", 'rb') as inp:
    a_values,results = pickle.load(inp)

plot_convergence_vs_parameter(
    results,
    title=r'Convergence comparison of RKHS–PI and SLSQP',
    filename='ConvergenceVsParameter',
    save=True,
    show=True
)