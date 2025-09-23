import numpy as np
from   functions import kernel, model,surrogate, observer, auxFunctions

# -------------------------------------------------------------------------------
# This script provides an example for generating a surrogate model for the value function
# of the Van der Pol system using the RKHS-PI method.
# -------------------------------------------------------------------------------

nMaxGreedy   = 1000
nMaxPI       = 10
newModel     = model.VanDerPol(1,0.1)
newOb        = observer.Observer("VanDerPol")
x            = np.linspace(-1,1,4)
X,Y          = np.meshgrid(x,x)
VFinput      = np.array([X.flatten(),Y.flatten()],dtype=np.float64)
VFoutputTrue = np.zeros(VFinput.shape[1])
for i in range(VFinput.shape[1]): # Compute the true value function for the input data using a BVP solver
    _,_,VF,_        = newModel.solveOLBVP(VFinput[:,i],50,1001)
    VFoutputTrue[i] = VF
x            = np.linspace(-1,1,100)
X,Y          = np.meshgrid(x,x)
testPoint    = np.array([X.flatten(),Y.flatten()],dtype=np.float64)

nMaxGreedy = 200
gamma      = 1.7
newKernel  = kernel.Gauss(gamma)
newSr      = surrogate.Surrogate(newKernel)
auxFunctions.RKHSPI(newModel,newSr,nMaxPI,nMaxGreedy,newOb,testPoint,VFinput,VFoutputTrue)
#auxFunctions.findBestGamma(newModel,newSr,nMaxGreedy,newOb,testPoint,[0.2,0.1,0.09,0.07,0.06,0.05,0.04,0.03,0.02,0.01])

nMaxGreedy = 200
gamma      = 1.1
newKernel  = kernel.GaussProduct(gamma,2)
newSr      = surrogate.SurrogateProductKernel(newKernel)
auxFunctions.RKHSPIProductKernel(newModel,newSr,nMaxPI,nMaxGreedy,newOb,testPoint,VFinput,VFoutputTrue)
#auxFunctions.findBestGammaProductKernel(newModel,newSr,nMaxGreedy,newOb,testPoint,[2.1,2,1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1,1.0])

newOb.plotObserver("VanDerPol")




