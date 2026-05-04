import numpy as np
from   functions import kernel, model,surrogate, observer, auxFunctions
import matplotlib.pyplot    as plt

# -------------------------------------------------------------------------------
# This script provides an example for generating a surrogate model for the value function
# of the academic toy example using the RKHS-PI method.
# -------------------------------------------------------------------------------

nMaxPI       = 10
newModel     = model.ToyExample()
newOb        = observer.Observer("ToyExample")
VFinput      = np.random.uniform(-1,1, size=(2, 100))
VFoutputTrue = newModel.trueVF(VFinput)
x            = np.linspace(-1,1,100)
X,Y          = np.meshgrid(x,x)
testPoint    = np.array([X.flatten(),Y.flatten()],dtype=np.float64)




nMaxGreedy = 200
gamma     = 1.3
newKernel = kernel.Gauss(gamma)
newSr     = surrogate.Surrogate(newKernel)
auxFunctions.RKHSPI(newModel,newSr,nMaxPI,nMaxGreedy,newOb,testPoint,VFinput,VFoutputTrue)
#auxFunctions.findBestGamma(newModel,newSr,nMaxGreedy,newOb,testPoint,[2.1,2,1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1,1.0])


nMaxGreedy = 200
gamma      = 1
newKernel  = kernel.GaussProduct(gamma,2)
newSr      = surrogate.SurrogateProductKernel(newKernel)
auxFunctions.RKHSPIProductKernel(newModel,newSr,nMaxPI,nMaxGreedy,newOb,testPoint,VFinput,VFoutputTrue)
#auxFunctions.findBestGammaProductKernel(newModel,newSr,nMaxGreedy,newOb,testPoint,[2.1,2,1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1,1.0,0.9,0.8,0.7,0.6,0.5])


newOb.plotObserver("ToyExample",0.5 * newModel.lambdaMin, 2   * newModel.lambdaMax )






