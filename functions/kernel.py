import numpy as np
import abc

# -----------------------------------------------------------------------------
# This class provides routines to
#   (i)  compute the generalized Gramian matrix used in the RKHS-PI method, and
#   (ii) evaluate the surrogate model and its gradient.
# Multiple kernel choices are supported.
# -----------------------------------------------------------------------------

class Kernel(metaclass=abc.ABCMeta):
    """
    Abstract base for radial kernels used in RKHS-PI constructions.
    kernels have the form k(x,y) = phi(||x-y||) 

    Subclasses must implement:
        - phi(r)   : radial basis function
        - phiR(r)  : phiR(r)  = d/dr phi(r)  * (1/r)
        - phiRR(r) : phiRR(r) = d/dr phiR(r) * (1/r)

    Parameters
    ----------
    gamma : float
        Kernel shape parameter.
    """
    def __init__(self,gamma):
        self.gamma    = gamma
    
    def setGamma(self,gamma):
        self.gamma = gamma

    # --- Abstract radial functions ----------------------------------------
    @abc.abstractmethod
    def phi(self,r): 
        pass    
    @abc.abstractmethod
    def phiR(self,r): 
        pass    
    @abc.abstractmethod
    def phiRR(self,r): 
        pass         

    # --- Gram matrix builders -----------------------------------------------------
    def getGramPDEStartColumn(self,f,y,fy = []): # Build the first column of a PDE Gram matrix using x = 0. 
        x     = np.atleast_2d(y[:,0]*0).T
        diff  = np.sqrt( np.abs(np.sum(x**2, axis=0, keepdims=True)  + np.sum(y**2, axis=0, keepdims=True).T  - 2 * y.T @ x ) )      
        phiR  = self.phiR(diff)       
            
        if len(fy)==0:
           fy = f(y)      
        return phiR * np.atleast_2d(np.sum(fy*y,axis=0)).T 

    def getGramPDE(self,f,y,x,fy = []): # Build the general PDE Gram block between y and x.
        diff  = np.sqrt( np.abs(np.sum(x**2, axis=0, keepdims=True)  + np.sum(y**2, axis=0, keepdims=True).T  - 2 * y.T @ x ) )      
        phiR  = self.phiR(diff)       
        phiRR = self.phiRR(diff)  
          
        fx    = f(x)
        if len(fy)==0:
            fy = f(y)
            
        term1 = phiRR*((np.sum(fy*y,axis=0) -x.T @ fy).T)*(np.sum(fx*x,axis=0) - y.T @ fx)
        term2 = (-1)*phiR* (fy.T@fx)

        return term1 + term2 
    
    # --- Evaluation helpers ------------------------------------------------
    def evalGrad(self,fx,y,x,alpha): # Evaluate the gradient of the kernel surrogate.
        diff  = np.sqrt( np.abs(np.sum(x**2, axis=0, keepdims=True)  + np.sum(y**2, axis=0, keepdims=True).T  - 2 * y.T @ x ) )      
        phiR  = self.phiR(diff)       
        phiRR = self.phiRR(diff)  
        alpha0 = alpha[0]
        alpha  = alpha[1:]

        term1 = phiRR* ( np.sum(fx*x,axis=0) - y.T @ fx  )  
        term3 = y * np.atleast_2d(term1 @ np.atleast_2d(alpha).T).T
        term4 = (-1)*(term1*np.atleast_2d(alpha) @ x.T).T
        term5 = (-1)*((phiR*np.atleast_2d(alpha)) @ fx.T).T

        x     = np.atleast_2d(x[:,0]*0).T
        diff  = np.sqrt( np.abs(np.sum(x**2, axis=0, keepdims=True)  + np.sum(y**2, axis=0, keepdims=True).T  - 2 * y.T @ x ) )      
        term6 = self.phiR(diff).T*y*alpha0   


        return term3+ term4+term5+ term6
    
    def evalFunc(self,fx,y,x,alpha): #Evaluate the the kernel surrogate.
        diff   = np.sqrt( np.abs(np.sum(x**2, axis=0, keepdims=True)  + np.sum(y**2, axis=0, keepdims=True).T  - 2 * y.T @ x ) )      
        phiR   = self.phiR(diff)    
        term2  = (phiR* ( np.sum(fx*x,axis=0) - y.T @ fx  ))
        x      = np.atleast_2d(x[:,0]*0).T
        diff   = np.sqrt( np.abs(np.sum(x**2, axis=0, keepdims=True)  + np.sum(y**2, axis=0, keepdims=True).T  - 2 * y.T @ x ) )     
        term1  = self.phi(diff)                 
        return  np.c_[term1,term2] @alpha 

# ---------------------------------------------------------------------------
# Concrete kernels
# ---------------------------------------------------------------------------

class QuadWendland(Kernel):
    def __init__(self,gamma,d):
        self.l        = np.floor(d/2)+ 2 +1 
        self.gamma    = gamma
    def phi(self,r):   return (self.gamma*r<=1) * (1-self.gamma*r)**(self.l+2) * ((self.l**2+4*self.l+3)*(self.gamma*r)**2+(3*self.l+6)*self.gamma*r+3)  
    def phiR(self,r):  return (self.gamma*r<=1) * (1-self.gamma*r)**(self.l+1) * (-self.gamma**2) * (12+7*self.l+self.l**2)*(1+(1+self.l)*self.gamma*r)
    def phiRR(self,r): return (self.gamma*r<=1) * (1-self.gamma*r)**(self.l) * self.gamma**4 * (24 + 50 * self.l + 35 *  self.l**2 + 10 *  self.l**3 + self.l**4) 

class QuadMatern(Kernel):
    def phi(self,r):   return np.exp(-self.gamma*r)*(3+3*self.gamma*r+self.gamma**2 *r**2)   
    def phiR(self,r):  return (-1)*np.exp(-self.gamma*r)*(1+self.gamma*r) * self.gamma**2 
    def phiRR(self,r): return self.gamma**4 * np.exp(-self.gamma*r) 

class Gauss(Kernel):
    def phi(self,r):   return np.exp(-(self.gamma*r)**2) 
    def phiR(self,r):  return (-2)*self.gamma**2 * np.exp(-(self.gamma*r)**2) 
    def phiRR(self,r): return 4*self.gamma**4 * np.exp(-(self.gamma*r)**2) 

class InvMulti(Kernel):
    def phi(self,r):   return 1/np.sqrt(1+self.gamma*(r**2)) 
    def phiR(self,r):  return (-1)*self.gamma* (1/np.sqrt((1+self.gamma*(r**2))**3)) 
    def phiRR(self,r): return 3*self.gamma**2 * (1/np.sqrt((1+self.gamma*(r**2))**5)) 

class LinMatern(Kernel):
    def phi(self,r):   return np.exp(-self.gamma*r)*(1+self.gamma*r)   
    def phiR(self,r):  return (-1)*np.exp(-self.gamma*r)*self.gamma**2 
    def phiRR(self,r): 
        diffMask1 = r<10**(-14)
        diffMask2 = r>10**(-14)
        return self.gamma**3 * np.exp(-self.gamma*r) * (1/(r+diffMask1)) * (diffMask2)
 


class KernelProduct(metaclass=abc.ABCMeta):
    """
    Like Kernel, but with an additional linear factor (y^T x)^d
    woven into the expressions; kernels have the form k(x,y) = phi(||x-y||) * (y^T x)^d;
    'case' controls which degree is used.

    Parameters
    ----------
    gamma : float
    case  : int
        0 -> no linear factor; 1 -> degree-1; >1 -> degree-d
    """
    def __init__(self,gamma,case = 0):
        self.case     = case
        self.gamma    = gamma
    
    def setGamma(self,gamma):
        self.gamma = gamma

    # --- Abstract radial functions ----------------------------------------
    @abc.abstractmethod
    def phi(self,r): 
        pass    
    @abc.abstractmethod
    def phiR(self,r): 
        pass    
    @abc.abstractmethod
    def phiRR(self,r): 
        pass         

    # --- Gram matrix builders -----------------------------------------------------
    def getGramPDE(self,f,y,x,fy = []): 
        if  self.case  >1:
            d     = self.case
            diff  = np.sqrt( np.abs(np.sum(x**2, axis=0, keepdims=True)  + np.sum(y**2, axis=0, keepdims=True).T  - 2 * y.T @ x ) )
            phi   = self.phi(diff)       
            phiR  = self.phiR(diff)       
            phiRR = self.phiRR(diff)  
            lin   = y.T@x

            fx    = f(x)
            if len(fy)==0:
               fy = f(y)

            term1     = phiRR*(lin**d)*((np.sum(fy*y,axis=0) -x.T @ fy).T)*(np.sum(fx*x,axis=0) - y.T @ fx)
            term2     = d*phiR*(lin**(d-1))*((np.sum(fy*y,axis=0)*0 +x.T @ fy).T)*(np.sum(fx*x,axis=0) - y.T @ fx)
            term3     = (-1)*phiR*(lin**d)*(fy.T@fx)
            term4     = d*phiR*(lin**(d-1))*((np.sum(fy*y,axis=0)-x.T @ fy).T)*(np.sum(fx*x,axis=0)*0 + y.T @ fx)
            term5     = d*(d-1)*phi*(lin**(d-2))*((np.sum(fy*y,axis=0)*0 +x.T @ fy).T)*(np.sum(fx*x,axis=0)*0 + y.T @ fx)
            term6     = d* phi *(lin**(d-1))*(fy.T@fx)
            
            return term1 + term2 + term3 + term4 + term5 + term6            
        elif self.case == 1:
            diff  = np.sqrt( np.abs(np.sum(x**2, axis=0, keepdims=True)  + np.sum(y**2, axis=0, keepdims=True).T  - 2 * y.T @ x ) )
            phi   = self.phi(diff)       
            phiR  = self.phiR(diff)       
            phiRR = self.phiRR(diff)  
            lin   = y.T@x

            fx    = f(x)
            if len(fy)==0:
               fy = f(y)

            term1     = phiRR*(lin)*((np.sum(fy*y,axis=0) -x.T @ fy).T)*(np.sum(fx*x,axis=0) - y.T @ fx)
            term2     = phiR * ((np.sum(fy*y,axis=0)*0 +x.T @ fy).T)*(np.sum(fx*x,axis=0) - y.T @ fx)
            term3     = (-1)*phiR*lin * (fy.T@fx)
            term4     = phiR*((np.sum(fy*y,axis=0) -x.T @ fy).T)*(np.sum(fx*x,axis=0)*0 + y.T @ fx)
            term5     = phi* (fy.T@fx)

            return term1 + term2 + term3 + term4 + term5 
        elif self.case == 0:    
            diff  = np.sqrt( np.abs(np.sum(x**2, axis=0, keepdims=True)  + np.sum(y**2, axis=0, keepdims=True).T  - 2 * y.T @ x ) )      
            phiR  = self.phiR(diff)       
            phiRR = self.phiRR(diff)  
            
            fx    = f(x)
            if len(fy)==0:
               fy = f(y)
            
            term1 = phiRR*((np.sum(fy*y,axis=0) -x.T @ fy).T)*(np.sum(fx*x,axis=0) - y.T @ fx)
            term2 = (-1)*phiR* (fy.T@fx)

            return term1 + term2 
        else:
            print("Case not implemented")    

    # --- Evaluation helpers ------------------------------------------------
    def evalGrad(self,fx,y,x,alpha): 
        if self.case > 1:
            d     = self.case
            diff  = np.sqrt( np.abs(np.sum(x**2, axis=0, keepdims=True)  + np.sum(y**2, axis=0, keepdims=True).T  - 2 * y.T @ x ) )
            phi   = self.phi(diff)       
            phiR  = self.phiR(diff)       
            phiRR = self.phiRR(diff)  
            lin   = y.T@x

            term10 = phiRR*(lin**d)*( np.sum(fx*x,axis=0) - y.T @ fx  ) 
            term11 = y * np.atleast_2d(term10 @ np.atleast_2d(alpha).T).T
            term12 = (-1)*(term10*np.atleast_2d(alpha) @ x.T).T

            term20 = d*phiR*(lin**(d-1))*( np.sum(fx*x,axis=0) - y.T @ fx  ) 
            term21 = (term20*np.atleast_2d(alpha) @ x.T).T

            term31 = ((((-1)*phiR*(lin**d))*np.atleast_2d(alpha)) @ fx.T).T

            term40 = d*phiR*(lin**(d-1))*( np.sum(fx*x,axis=0)*0 + y.T @ fx  )  
            term41 = y * np.atleast_2d(term40 @ np.atleast_2d(alpha).T).T
            term42 = (-1)*(term40*np.atleast_2d(alpha) @ x.T).T

            term50 = d*(d-1)*phi*(lin**(d-2))*( np.sum(fx*x,axis=0)*0 + y.T @ fx  )  
            term51 = (term50*np.atleast_2d(alpha) @ x.T).T

            term61 = (((d* phi *(lin**(d-1)))*np.atleast_2d(alpha)) @ fx.T).T

            return term11+term12+term21+term31+term41+term42+term51+term61             
        elif self.case == 1:
            diff  = np.sqrt( np.abs(np.sum(x**2, axis=0, keepdims=True)  + np.sum(y**2, axis=0, keepdims=True).T  - 2 * y.T @ x ) )      
            phi   = self.phi(diff)             
            phiR  = self.phiR(diff)       
            phiRR = self.phiRR(diff)  
            lin   = y.T@x

            termTemp1 = phiRR*lin* ( np.sum(fx*x,axis=0) - y.T @ fx  )  
            term1     = y * np.atleast_2d(termTemp1 @ np.atleast_2d(alpha).T).T
            term2     = (-1)*(termTemp1*np.atleast_2d(alpha) @ x.T).T

            termTemp2 = phiR*( np.sum(fx*x,axis=0) - y.T @ fx  )  
            term3     = (termTemp2*np.atleast_2d(alpha) @ x.T).T

            term4     = (-1)*(((phiR*lin)*np.atleast_2d(alpha)) @ fx.T).T

            termTemp3 = phiR* ( np.sum(fx*x,axis=0)*0 + y.T @ fx  )  
            term5     = y * np.atleast_2d(termTemp3 @ np.atleast_2d(alpha).T).T
            term6     = (-1)*(termTemp3*np.atleast_2d(alpha) @ x.T).T

            term7     = (((phi)*np.atleast_2d(alpha)) @ fx.T).T

            return term1+ term2+term3+term4+ term5+term6+term7

        elif self.case == 0:   
            diff  = np.sqrt( np.abs(np.sum(x**2, axis=0, keepdims=True)  + np.sum(y**2, axis=0, keepdims=True).T  - 2 * y.T @ x ) )      
            phiR  = self.phiR(diff)       
            phiRR = self.phiRR(diff)  

            term1 = phiRR* ( np.sum(fx*x,axis=0) - y.T @ fx  )  
            term3 = y * np.atleast_2d(term1 @ np.atleast_2d(alpha).T).T
            term4 = (-1)*(term1*np.atleast_2d(alpha) @ x.T).T
            term5 = (-1)*((phiR*np.atleast_2d(alpha)) @ fx.T).T

            return term3+ term4+term5
        else:
            print("Case not implemented")     

    def evalFunc(self,fx,y,x,alpha):
        if self.case > 1:
            d     = self.case
            diff  = np.sqrt( np.abs(np.sum(x**2, axis=0, keepdims=True)  + np.sum(y**2, axis=0, keepdims=True).T  - 2 * y.T @ x ) )      
            phi   = self.phi(diff)             
            phiR  = self.phiR(diff)   
            lin   = y.T@x  
            return (phiR*(lin**d) *  ( np.sum(fx*x,axis=0) - y.T @ fx  ))@alpha + d*(phi*(lin**(d-1))* ( np.sum(fx*x,axis=0)*0 + y.T @ fx  ))@alpha                 
        elif self.case == 1:
            diff  = np.sqrt( np.abs(np.sum(x**2, axis=0, keepdims=True)  + np.sum(y**2, axis=0, keepdims=True).T  - 2 * y.T @ x ) )      
            phi   = self.phi(diff)             
            phiR  = self.phiR(diff)   
            lin   = y.T@x  
            return (phiR*lin *  ( np.sum(fx*x,axis=0) - y.T @ fx  ))@alpha + (phi* ( np.sum(fx*x,axis=0)*0 + y.T @ fx  ))@alpha         
        elif self.case == 0:
            diff  = np.sqrt( np.abs(np.sum(x**2, axis=0, keepdims=True)  + np.sum(y**2, axis=0, keepdims=True).T  - 2 * y.T @ x ) )      
            phiR  = self.phiR(diff)    
            return (phiR* ( np.sum(fx*x,axis=0) - y.T @ fx  ))@alpha
        else:
            print("Case not implemented")    

# ---------------------------------------------------------------------------
# Concrete product kernels
# ---------------------------------------------------------------------------

class QuadWendlandProduct(KernelProduct):
    def __init__(self,gamma,d,case = 0):
        self.l        = np.floor(d/2)+ 2 +1 
        self.case     = case
        self.gamma    = gamma
    def phi(self,r):   return (self.gamma*r<=1) * (1-self.gamma*r)**(self.l+2) * ((self.l**2+4*self.l+3)*(self.gamma*r)**2+(3*self.l+6)*self.gamma*r+3)  
    def phiR(self,r):  return (self.gamma*r<=1) * (1-self.gamma*r)**(self.l+1) * (-self.gamma**2) * (12+7*self.l+self.l**2)*(1+(1+self.l)*self.gamma*r)
    def phiRR(self,r): return (self.gamma*r<=1) * (1-self.gamma*r)**(self.l) * self.gamma**4 * (24 + 50 * self.l + 35 *  self.l**2 + 10 *  self.l**3 + self.l**4) 

class QuadMaternProduct(KernelProduct):
    def phi(self,r):   return np.exp(-self.gamma*r)*(3+3*self.gamma*r+self.gamma**2 *r**2)   
    def phiR(self,r):  return (-1)*np.exp(-self.gamma*r)*(1+self.gamma*r) * self.gamma**2 
    def phiRR(self,r): return self.gamma**4 * np.exp(-self.gamma*r) 

class GaussProduct(KernelProduct):
    def phi(self,r):   return np.exp(-(self.gamma*r)**2) 
    def phiR(self,r):  return (-2)*self.gamma**2 * np.exp(-(self.gamma*r)**2) 
    def phiRR(self,r): return 4*self.gamma**4 * np.exp(-(self.gamma*r)**2) 

class InvMultiProduct(KernelProduct):
    def phi(self,r):   return 1/np.sqrt(1+self.gamma*(r**2)) 
    def phiR(self,r):  return (-1)*self.gamma* (1/np.sqrt((1+self.gamma*(r**2))**3)) 
    def phiRR(self,r): return 3*self.gamma**2 * (1/np.sqrt((1+self.gamma*(r**2))**5)) 

class LinMaternProduct(KernelProduct):
    def phi(self,r):   return np.exp(-self.gamma*r)*(1+self.gamma*r)   
    def phiR(self,r):  return (-1)*np.exp(-self.gamma*r)*self.gamma**2 
    def phiRR(self,r): 
        diffMask1 = r<10**(-14)
        diffMask2 = r>10**(-14)
        return self.gamma**3 * np.exp(-self.gamma*r) * (1/(r+diffMask1)) * (diffMask2)



