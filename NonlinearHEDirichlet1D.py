import numpy as np
from   functions import kernel, model,surrogate, observer, auxFunctions

# -------------------------------------------------------------------------------
# This script provides an example for generating a surrogate model for the value function
# of the nonlinear heat equation problem (\beta=1) with Dirichlet boundary conditions using the RKHS-PI method.
# -------------------------------------------------------------------------------

nMaxPI       = 10
N            = 50
newModel     = model.HEDirichlet1D(N,[1,1,1,0.01,3000])
newOb        = observer.Observer("NonlinearHEDirichlet1D")
VFinput      = (np.random.uniform(0,1, size=(N, 100))-0.5)*20
VFoutputTrue = np.zeros(VFinput.shape[1])
testPoint    = (np.random.uniform(0,1, size=(N, 10**5))-0.5)*20
for i in range(VFinput.shape[1]): # Compute the true value function for the input data using a BVP solver
    _,_,VF,_        = newModel.solveOLIterativ(VFinput[:,i],50,10001)
    VFoutputTrue[i] = VF

nMaxGreedy = 2000
gamma      = 3 * 10**(-5)
newKernel  = kernel.Gauss(gamma)
newSr      = surrogate.Surrogate(newKernel)
auxFunctions.RKHSPI(newModel,newSr,nMaxPI,nMaxGreedy,newOb,testPoint,VFinput,VFoutputTrue)
#auxFunctions.findBestGamma(newModel,newSr,nMaxGreedy,newOb,testPoint,[6*10**(-10)])

nMaxGreedy = 2000
gamma      = 4*10**(-8)
newKernel  = kernel.LinMaternProduct(gamma,2)
newSr      = surrogate.SurrogateProductKernel(newKernel)
auxFunctions.RKHSPIProductKernel(newModel,newSr,nMaxPI,nMaxGreedy,newOb,testPoint,VFinput,VFoutputTrue)
# #auxFunctions.findBestGammaProductKernel(newModel,newSr,nMaxGreedy,newOb,testPoint,[4*10**(-8),3*10**(-8),2*10**(-8)])

newOb.plotObserver("NonlinearHEDirichlet1D",0.5 * newModel.lambdaMin, 2   * newModel.lambdaMax )











