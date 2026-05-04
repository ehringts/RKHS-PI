import numpy as np
import numpy.matlib
import pickle
import time
import scipy
from scipy.linalg import cholesky
import matplotlib.pyplot    as plt
import copy
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'   # or 'stix' for STIX fonts
mpl.rcParams['font.family'] = 'serif'
from matplotlib.ticker import ScalarFormatter, MaxNLocator, FuncFormatter

# -----------------------------------------------------------------------------
# This class provides routines to
#   (i) monitor the progress of the RKHS-PI algorithm
#   (ii) plot the results 
# -----------------------------------------------------------------------------


class Observer(): # Observer class to monitor the progress of the RKHS-PI algorithm
    def __init__(self,name):
        self.trueErrorListStandart        = []
        self.performanceListStandart      = []
        self.resErrorListStandart         = []
        self.stagListStandart             = []
        self.GreedyErrorListStandart      = []
        self.trueErrorListQuadKernel      = []
        self.performanceListQuadKernel    = []
        self.resErrorListQuadKernel       = []
        self.stagListQuadKernel           = []
        self.GreedyErrorListQuadKernel    = []
        self.upperListQuadKernel          = []
        self.lowerListQuadKernel          = []
        self.upperListStandart            = []
        self.lowerListStandart            = []
        self.name                         = name
    
    def saveObserver(self,name): # Save the observer to a file
        with open("data/"+name, 'wb') as outp:
             pickle.dump(self, outp, protocol=4)

    def loadObserver(self,name): # Load the observer from a file
        with open("data/"+name, 'rb') as inp:
             oldSelf                           = pickle.load(inp)
             self.trueErrorListStandart        = oldSelf.trueErrorListStandart 
             self.performanceListStandart      = oldSelf.performanceListStandart 
             self.resErrorListStandart         = oldSelf.resErrorListStandart 
             self.GreedyErrorListStandart      = oldSelf.GreedyErrorListStandart 
             self.stagListStandart             = oldSelf.stagListStandart 
             self.trueErrorListQuadKernel      = oldSelf.trueErrorListQuadKernel
             self.performanceListQuadKernel    = oldSelf.performanceListQuadKernel
             self.resErrorListQuadKernel       = oldSelf.resErrorListQuadKernel
             self.GreedyErrorListQuadKernel    = oldSelf.GreedyErrorListQuadKernel
             self.stagListQuadKernel           = oldSelf.stagListQuadKernel     

             self.upperListQuadKernel          = oldSelf.upperListQuadKernel
             self.lowerListQuadKernel          = oldSelf.lowerListQuadKernel
             self.upperListStandart            = oldSelf.upperListStandart
             self.lowerListStandart            = oldSelf.lowerListStandart     

             self.name                         = oldSelf.name

    def addObjectGreedyErrorStandart(self,GreedyError):
        self.GreedyErrorListStandart.append(copy.copy(GreedyError))
        if self.name:
           self.saveObserver(self.name) 
    def addObjectGreedyErrorQuadKernel(self,GreedyError):
        self.GreedyErrorListQuadKernel.append(copy.copy(GreedyError))
        if self.name:
           self.saveObserver(self.name) 


    def addObjectStandart(self,trueError,performance,resError,stag):
        self.trueErrorListStandart.append(copy.copy(trueError))
        self.performanceListStandart.append(copy.copy(performance))
        self.resErrorListStandart.append(copy.copy(resError))
        self.stagListStandart.append(copy.copy(stag))
        if self.name:
           self.saveObserver(self.name) 
    def addObjectQuadKernel(self,trueError,performance,resError,stag):
        self.trueErrorListQuadKernel.append(copy.copy(trueError))
        self.performanceListQuadKernel.append(copy.copy(performance))
        self.resErrorListQuadKernel.append(copy.copy(resError))
        self.stagListQuadKernel.append(copy.copy(stag))
        if self.name:
           self.saveObserver(self.name) 

    def addQuadraticStandart(self,lower,upper):
        self.lowerListStandart.append(copy.copy(lower))
        self.upperListStandart.append(copy.copy(upper))
        if self.name:
           self.saveObserver(self.name) 

    def addQuadraticQuadKernel(self,lower,upper):
        self.lowerListQuadKernel.append(copy.copy(lower))
        self.upperListQuadKernel.append(copy.copy(upper))
        if self.name:
           self.saveObserver(self.name) 


    def plotObserver(self, title, lower, upper):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, FuncFormatter

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

            color_std = '#1f77b4'
            color_quad = '#ff7f0e'
            color_ref = 'black'

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                2, 2,
                figsize=(12, 5.5),
                gridspec_kw={'height_ratios': [1, 0.33]}
            )
            fig.subplots_adjust(
                left=0.06,
                right=0.995,
                bottom=0.10,
                top=0.93,
                wspace=0.22,
                hspace=0.35
            )

            def set_zoomed_yaxis(ax, *arrays, pad_ratio=0.15, decimal=False):
                y_all = []
                for arr in arrays:
                    arr = np.asarray(arr, dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size > 0:
                        y_all.append(arr)

                if not y_all:
                    return

                y = np.concatenate(y_all)
                ymin = np.min(y)
                ymax = np.max(y)

                if np.isclose(ymin, ymax):
                    pad = max(abs(ymin) * 1e-6, 1e-12)
                else:
                    pad = (ymax - ymin) * pad_ratio

                ax.set_ylim(ymin - pad, ymax + pad)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

                if decimal:
                    ax.yaxis.set_major_formatter(
                        FuncFormatter(lambda x, pos: f"{x:.2f}")
                    )
                else:
                    ax.yaxis.set_major_formatter(
                        FuncFormatter(lambda x, pos: f"{x:.2e}")
                    )

            # -------------------------------------------------
            # 1) Greedy residual
            # -------------------------------------------------
            x_std_greedy = np.arange(1, len(self.GreedyErrorListStandart))
            x_quad_greedy = np.arange(1, len(self.GreedyErrorListQuadKernel))

            ax1.semilogy(
                x_std_greedy,
                self.GreedyErrorListStandart[1:],
                color=color_std,
                linestyle='-',
                linewidth=wid,
                label=r'standard kernel'
            )
            ax1.semilogy(
                x_quad_greedy,
                self.GreedyErrorListQuadKernel[1:],
                color=color_quad,
                linestyle='--',
                linewidth=wid,
                label=r'product kernel'
            )

            ax1.set_xlim(
                1,
                max(len(self.GreedyErrorListStandart), len(self.GreedyErrorListQuadKernel)) - 1
            )
            ax1.set_title(r'$\mathrm{ResidualGHJB}$')
            ax1.set_xlabel(r'\# greedy iterations')
            ax1.set_ylabel(r'residual')
            ax1.legend(loc='upper right', frameon=True)
            ax1.grid(True)

            # -------------------------------------------------
            # 2) PI error
            # -------------------------------------------------
            x_std_true = np.arange(len(self.trueErrorListStandart))
            x_quad_true = np.arange(len(self.trueErrorListQuadKernel))

            ax2.semilogy(
                x_std_true,
                self.trueErrorListStandart,
                color=color_std,
                linestyle='-',
                linewidth=wid
            )
            ax2.semilogy(
                x_quad_true,
                self.trueErrorListQuadKernel,
                color=color_quad,
                linestyle='--',
                linewidth=wid
            )

            ax2.set_xlim(
                0,
                max(len(self.trueErrorListStandart), len(self.trueErrorListQuadKernel)) - 1
            )
            ax2.set_title(r'$\mathrm{TrueError}$')
            ax2.set_xlabel(r'\# PI iterations')
            ax2.set_ylabel(r'error')
            ax2.grid(True)

            # -------------------------------------------------
            # 3) Lower spectral quantity
            # -------------------------------------------------
            x_std_lower = np.arange(len(self.lowerListStandart))
            x_quad_lower = np.arange(len(self.lowerListQuadKernel))

            y_std_lower = np.asarray(self.lowerListStandart, dtype=float)
            y_quad_lower = np.asarray(self.lowerListQuadKernel, dtype=float)

            ax3.plot(
                x_std_lower,
                y_std_lower,
                color=color_std,
                linestyle='-',
                linewidth=wid
            )
            ax3.plot(
                x_quad_lower,
                y_quad_lower,
                color=color_quad,
                linestyle='--',
                linewidth=wid
            )
            ax3.axhline(
                y=lower,
                color=color_ref,
                linestyle=':',
                linewidth=1.4,
                label=r'$\frac{1}{2}\lambda_{\min}(P)$'
            )

            ax3.set_xlim(
                0,
                max(len(self.lowerListStandart), len(self.lowerListQuadKernel)) - 1
            )
            set_zoomed_yaxis(ax3, y_std_lower, y_quad_lower, [lower], decimal=True)
            ax3.set_title(r'$ \mathrm{MinQuadraticBound}$')
            ax3.set_xlabel(r'\# PI iterations')
            ax3.set_ylabel(r'value')
            ax3.legend(loc='center right', frameon=True)
            ax3.grid(True)

            # -------------------------------------------------
            # 4) Upper spectral quantity
            # -------------------------------------------------
            x_std_upper = np.arange(len(self.upperListStandart))
            x_quad_upper = np.arange(len(self.upperListQuadKernel))

            y_std_upper = np.asarray(self.upperListStandart, dtype=float)
            y_quad_upper = np.asarray(self.upperListQuadKernel, dtype=float)

            ax4.plot(
                x_std_upper,
                y_std_upper,
                color=color_std,
                linestyle='-',
                linewidth=wid
            )
            ax4.plot(
                x_quad_upper,
                y_quad_upper,
                color=color_quad,
                linestyle='--',
                linewidth=wid
            )
            ax4.axhline(
                y=upper,
                color=color_ref,
                linestyle=':',
                linewidth=1.4,
                label=r'$2\lambda_{\max}(P)$'
            )

            ax4.set_xlim(
                0,
                max(len(self.upperListStandart), len(self.upperListQuadKernel)) - 1
            )
            set_zoomed_yaxis(ax4, y_std_upper, y_quad_upper, [upper], decimal=True)
            ax4.set_title(r'$\mathrm{MaxQuadraticBound}$')
            ax4.set_xlabel(r'\# PI iterations')
            ax4.set_ylabel(r'value')
            ax4.legend(loc='center right', frameon=True)
            ax4.grid(True)

            plt.tight_layout()
            plt.savefig(f"{title}.pdf", bbox_inches='tight',pad_inches=0.02, dpi=1200)
            plt.show()