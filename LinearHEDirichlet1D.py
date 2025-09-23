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
testPoint    = np.random.uniform(0,1, size=(N, 20000))*10
VFinput      = testPoint
VFoutputTrue = newModel.trueVF(VFinput)

nMaxGreedy = 2000
gamma      = 6*10**(-10)
newKernel  = kernel.Gauss(gamma)
newSr      = surrogate.Surrogate(newKernel)
auxFunctions.RKHSPI(newModel,newSr,nMaxPI,nMaxGreedy,newOb,testPoint,VFinput,VFoutputTrue)
#auxFunctions.findBestGamma(newModel,newSr,nMaxGreedy,newOb,testPoint,[6*10**(-10)])

nMaxGreedy = 2000
gamma      = 5*10**(-8)
newKernel  = kernel.LinMaternProduct(gamma,2)
newSr      = surrogate.SurrogateProductKernel(newKernel)
auxFunctions.RKHSPIProductKernel(newModel,newSr,nMaxPI,nMaxGreedy,newOb,testPoint,VFinput,VFoutputTrue)
#auxFunctions.findBestGammaProductKernel(newModel,newSr,nMaxGreedy,newOb,testPoint,[2.1,2,1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1,1.0])

newOb.plotObserver("LinearHEDirichlet1D")



