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
        self.name                         = name
    
    def saveObserver(self,name): # Save the observer to a file
        with open("data/"+name, 'wb') as outp:
             pickle.dump(self, outp, protocol=4)

    def loadObserver(self,name): # Load the observer from a file
        with open("data/"+name, 'rb') as inp:
             oldSelf               = pickle.load(inp)
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
             self.name                      = oldSelf.name

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


    def plotObserver(self,titel): # Plot the results stored in the observer
        wid = 1.1
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.semilogy(np.linspace(1,len(self.GreedyErrorListStandart),len(self.GreedyErrorListStandart)),self.GreedyErrorListStandart, 'y',linewidth=wid, label=r'$\mathrm{Res\text{-}GHJB}$, $k_{G}$, $\gamma=\sqrt{6} \cdot 10^{-5}  $')
        ax1.semilogy(np.linspace(1,len(self.GreedyErrorListQuadKernel),len(self.GreedyErrorListQuadKernel)),self.GreedyErrorListQuadKernel, 'r',linewidth=wid, label=r'$\mathrm{Res\text{-}GHJB}$, $k_{LM,Q}$, $\gamma=4 \cdot 10^{-8} $')
        ax1.set_xlim(1, len(self.GreedyErrorListStandart))
        ax1.set_xlabel('# greedy iterations',loc='center')
        ax1.legend()

        ax1.grid(visible=True)

        ax2.semilogy(np.linspace(1,len(self.trueErrorListStandart),len(self.trueErrorListStandart)),self.trueErrorListStandart, 'y' ,linewidth=wid, label=r'$\mathrm{Error\text{-}PI}$, $k_{G}$,  $\gamma=\sqrt{6} \cdot 10^{-5}  $')
        ax2.semilogy(np.linspace(1,len(self.trueErrorListQuadKernel),len(self.trueErrorListQuadKernel)),self.trueErrorListQuadKernel, 'r',linewidth=wid, label=r'$\mathrm{Error\text{-}PI}$,$k_{LM,Q}$, $\gamma=4 \cdot 10^{-8} $')
       
        ax2 = plt.gca()
        ax2.set_xlim(1, len(self.trueErrorListStandart))
        ax2.set_xlabel('# PI iterations',loc='center')

        ax2.legend()
        ax2.grid(visible=True)
        plt.subplots_adjust(wspace=0.15)
        plt.savefig(str(titel)+'.pdf', bbox_inches='tight', dpi=1200)  
        plt.show()


