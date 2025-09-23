import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------------------------------------------------------------
# This functions implements the RKHS-PI algorithm and routines to find the best
# parameter gamma for the kernels used in the surrogate model
# -----------------------------------------------------------------------------


def RKHSPI(model,surrogate,nMaxPI,nMaxGreedy,observer,testPoint,VFinput,VFoutputTrue): # RKHS-PI algorithm
    for j in range(nMaxPI):
        if  j==0:
            mu              = lambda x: model.stableControl(x)
            F               = lambda x: model.getF(x,mu(x))
            rhs             = lambda x: model.getRHS(x,mu(x))
            surrogate.doFGreedy(F,rhs,testPoint,nMaxGreedy,observer,10**(-8))
            maxFGreedyError = surrogate.trainError[-1]
            oldSRVal        = 0
        else:
            mu              = lambda x: model.getMuFromSr(x,surrogate.kernel.evalGrad(FCenterVal,x,center,alpha))
            F               = lambda x: model.getF(x,mu(x))
            rhs             = lambda x: model.getRHS(x,mu(x)) 
            surrogate.fit(F,rhs,surrogate.center, iterativ= False)
            maxFGreedyError = np.max(np.abs(np.c_[surrogate.kernel.getGramPDEStartColumn(F,testPoint),surrogate.kernel.getGramPDE(F,testPoint,surrogate.center)]@surrogate.alpha-rhs(testPoint))/(np.abs(rhs(testPoint))+10**(-8)))
        FCenterVal = F(surrogate.center).copy()
        center     = surrogate.center.copy()
        alpha      = surrogate.alpha.copy()
        trueError  = np.sqrt( np.sum(np.abs(VFoutputTrue-surrogate.kernel.evalFunc(FCenterVal,VFinput,center,alpha))**2)/np.sum(np.abs(VFoutputTrue)**2))
        evalSR     = surrogate.kernel.evalFunc(FCenterVal,testPoint,center,alpha)  
        isPos         = np.min(evalSR)>=0
        hasLowerBound = np.min(evalSR - 0.5 * model.lambdaMin * np.sum(testPoint**2,0) ) >= 0
        hasUpperBound = np.max(evalSR - 2   * model.lambdaMax * np.sum(testPoint**2,0) ) <= 0
        print(str(j) + " Iteration: True-Error = " + str( trueError ) + ", Residual-Error = " + str(maxFGreedyError) +", stagnation-Error = " + str( np.max(np.abs(oldSRVal-surrogate.kernel.evalFunc(FCenterVal,testPoint,center,alpha))) )+", is positive = " + str(isPos)+", Lower Bound = " + str(hasLowerBound)+", Upper Bound = " + str(hasUpperBound))
        observer.addObjectStandart(trueError,0,maxFGreedyError,np.max(np.abs(oldSRVal-surrogate.kernel.evalFunc(FCenterVal,testPoint,center,alpha))))
        oldSRVal = surrogate.kernel.evalFunc(FCenterVal,testPoint,center,alpha)    
       
def findBestGamma(model,surrogate,nMaxGreedy,observer,testPoint,gammaList): 
    gammaListVal    = 1000+np.zeros(len(gammaList))
    for i in range(len(gammaList)):
        surrogate.kernel.setGamma(gammaList[i])
        mu              = lambda x: model.stableControl(x)
        F               = lambda x: model.getF(x,mu(x))
        rhs             = lambda x: model.getRHS(x,mu(x))
        surrogate.doFGreedy(F,rhs,testPoint,nMaxGreedy,observer,10**(-8))
        surrogate.C     = np.array([])
        gammaListVal[i] = surrogate.trainError[-1]
        bestGammaIndex  = np.argmin(gammaListVal)
        print("Gamma: " + str(gammaList[i]) + ", Error: " + str(gammaListVal[i]))
        print("The current best Gamma is: " + str(gammaList[bestGammaIndex]) + " with a value of " + str(gammaListVal[bestGammaIndex]))
    bestGammaIndex = np.argmin(gammaListVal)
    print("The best Gamma is: " + str(gammaList[bestGammaIndex]) + " with a value of " + str(gammaListVal[bestGammaIndex]))
    return gammaList[bestGammaIndex], gammaListVal[bestGammaIndex] 
            

def RKHSPIProductKernel(model,surrogate,nMaxPI,nMaxGreedy,observer,testPoint,VFinput,VFoutputTrue): # RKHS-PI algorithm for product kernels
    for j in range(nMaxPI):
        if  j==0:
            mu              = lambda x: model.stableControl(x)
            F               = lambda x: model.getF(x,mu(x))
            rhs             = lambda x: model.getRHS(x,mu(x))
            surrogate.doFGreedy(F,rhs,testPoint,nMaxGreedy,observer,10**(-8))
            maxFGreedyError = surrogate.trainError[-1]
            oldSRVal        = 0
        else:
            mu              = lambda x: model.getMuFromSr(x,surrogate.kernel.evalGrad(FCenterVal,x,center,alpha))
            F               = lambda x: model.getF(x,mu(x))
            rhs             = lambda x: model.getRHS(x,mu(x)) 
            surrogate.fit(F,rhs,surrogate.center, iterativ= False)
            maxFGreedyError = np.max(np.abs(surrogate.kernel.getGramPDE(F,testPoint,surrogate.center)@surrogate.alpha-rhs(testPoint))/(np.abs(rhs(testPoint))+10**(-8)))
        FCenterVal = F(surrogate.center).copy()
        center     = surrogate.center.copy()
        alpha      = surrogate.alpha.copy()
        trueError  = np.sqrt( np.sum(np.abs(VFoutputTrue-surrogate.kernel.evalFunc(FCenterVal,VFinput,center,alpha))**2)/np.sum(np.abs(VFoutputTrue)**2))
        evalSR     = surrogate.kernel.evalFunc(FCenterVal,testPoint,center,alpha)  
        isPos         = np.min(evalSR)>=0
        hasLowerBound = np.min(evalSR - 0.5 * model.lambdaMin * np.sum(testPoint**2,0) ) >= 0
        hasUpperBound = np.max(evalSR - 2   * model.lambdaMax * np.sum(testPoint**2,0) ) <= 0
        print(str(j) + " Iteration: True-Error = " + str( trueError ) + ", Residual-Error = " + str(maxFGreedyError) +", stagnation-Error = " + str( np.max(np.abs(oldSRVal-surrogate.kernel.evalFunc(FCenterVal,testPoint,center,alpha))) )+", is positive = " + str(isPos)+", Lower Bound = " + str(hasLowerBound)+", Upper Bound = " + str(hasUpperBound))
        observer.addObjectQuadKernel(trueError,0,maxFGreedyError,np.max(np.abs(oldSRVal-surrogate.kernel.evalFunc(FCenterVal,testPoint,center,alpha))))
        oldSRVal = surrogate.kernel.evalFunc(FCenterVal,testPoint,center,alpha)    

def findBestGammaProductKernel(model,surrogate,nMaxGreedy,observer,testPoint,gammaList): # Find the best gamma for product kernels
    gammaListVal    = 1000+np.zeros(len(gammaList))
    for i in range(len(gammaList)):
        surrogate.kernel.setGamma(gammaList[i])
        mu              = lambda x: model.stableControl(x)
        F               = lambda x: model.getF(x,mu(x))
        rhs             = lambda x: model.getRHS(x,mu(x))
        surrogate.doFGreedy(F,rhs,testPoint,nMaxGreedy,observer,10**(-8))
        surrogate.C     = np.array([])
        gammaListVal[i] = surrogate.trainError[-1]
        bestGammaIndex  = np.argmin(gammaListVal)
        print("Gamma: " + str(gammaList[i]) + ", Error: " + str(gammaListVal[i]))
        print("The current best Gamma is: " + str(gammaList[bestGammaIndex]) + " with a value of " + str(gammaListVal[bestGammaIndex]))
    bestGammaIndex = np.argmin(gammaListVal)
    print("The best Gamma is: " + str(gammaList[bestGammaIndex]) + " with a value of " + str(gammaListVal[bestGammaIndex]))
    return gammaList[bestGammaIndex], gammaListVal[bestGammaIndex] 
