import numpy as np
from   functions import kernel, model,surrogate, observer, auxFunctions

# -------------------------------------------------------------------------------
# This script provides an example for generating a surrogate model for the value function
# of the linear heat equation problem (\beta=0) with Dirichlet boundary conditions using the RKHS-PI method.
# -------------------------------------------------------------------------------

nMaxPI       = 10
N            = 50
newModel     = model.HEDirichlet1D(N,[1,0,1,0.01,3000])
newOb        = observer.Observer("LinearHEDirichlet1D")
testPoint    = (np.random.uniform(0,1, size=(N, 10**5))-0.5)*20
VFinput      = (np.random.uniform(0,1, size=(N, 100))-0.5)*20
VFoutputTrue = newModel.trueVF(VFinput)

nMaxGreedy = 2000
gamma      = 2 * 10**(-5)
newKernel  = kernel.Gauss(gamma)
newSr      = surrogate.Surrogate(newKernel)
auxFunctions.RKHSPI(newModel,newSr,nMaxPI,nMaxGreedy,newOb,testPoint,VFinput,VFoutputTrue)
#auxFunctions.findBestGamma(newModel,newSr,nMaxGreedy,newOb,testPoint,[2 * 10**(-5),1 * 10**(-5),9 * 10**(-6),8 * 10**(-6),7 * 10**(-6),6 * 10**(-6),5 * 10**(-6),4 * 10**(-6),3 * 10**(-6),2.5 * 10**(-6),2 * 10**(-6),1.5 * 10**(-6),1 * 10**(-6)])



# nMaxGreedy = 2000
# gamma      = 1 * 10**(-8)
# newKernel  = kernel.LinMaternProduct(gamma,2)
# newSr      = surrogate.SurrogateProductKernel(newKernel)
# auxFunctions.RKHSPIProductKernel(newModel,newSr,nMaxPI,nMaxGreedy,newOb,testPoint,VFinput,VFoutputTrue)
# #auxFunctions.findBestGammaProductKernel(newModel,newSr,nMaxGreedy,newOb,testPoint,[5 * 10**(-8),4 * 10**(-8),3 * 10**(-8),2 * 10**(-8),1 * 10**(-8)])

# newOb.plotObserver("LinearHEDirichlet1D",0.5 * newModel.lambdaMin, 2   * newModel.lambdaMax )



