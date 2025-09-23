import numpy                as np
from   scipy.integrate      import solve_bvp
import matplotlib.pyplot    as plt
import time
from   scipy.interpolate    import interp1d
from   scipy.integrate      import solve_ivp, quad
import matplotlib.animation as animation
import abc
from scipy                  import linalg as la

# -----------------------------------------------------------------------------
# This class provides routines to
#   (i)  implement optimal control problems of the form int_0^\infty h(x) + u^\top R u dt subject to x'=f(x)+g(x)u, x(0)=x0
#   (ii) solve the corresponding open-loop boundary value problem
# -----------------------------------------------------------------------------


class Model(metaclass=abc.ABCMeta):
    def __init__(self, stateWeight, controlWeight):
        self.stateWeight   = stateWeight    # Weight in the cost functional for the state, i.e. h(x) = stateWeight * ||x||^2
        self.controlWeight = controlWeight  # Weight in the cost functional for the control, i.e. R = controlWeight * I

    @abc.abstractmethod
    def f(self,x): 
        pass    
    @abc.abstractmethod
    def g(self,x): 
        pass      
    @abc.abstractmethod
    def jacobi_of_f_transposed_dot_p(self,x,p): # Jacobian of f(x) transposed, times p
        pass     

    def solveOLBVP(self,startState,endT,numbOfEval): # Solve open-loop optimal control problem as a boundary value problem
        n = startState.shape[0]
        def fun(t, y):  
            x  = y[:n,:]
            p  = y[n:2*n,:]
            u  = (-1/(2*self.controlWeight)) * self.g(x).T @ p
            A1 = self.f(x)+self.g(x) @ u
            A2 =-self.jacobi_of_f_transposed_dot_p(x,p)-2*x * self.stateWeight
            A3 = np.atleast_2d(-self.stateWeight * np.sum(x**2,0) - self.controlWeight * np.sum(u**2,0))
            return  np.r_[A1, A2,A3]        
        def bc(ya, yb): 
            return  np.r_[ya[0:n]-startState,yb[n:2*n+1]]
        ti      = time.time()
        tSpan   = np.linspace(0,endT ,numbOfEval) 
        initSol = np.zeros((2*n+1,tSpan.size))
        sol     = solve_bvp(fun, bc, tSpan, initSol,max_nodes = 10000000, tol = 1e-8)
        print("Solved open-loop control with " + str(sol.x.shape[0]) + " mesh points, maximal residual error of " + str(np.amax(np.abs(sol.rms_residuals))) + ". It took "+str(time.time()-ti)+" seconds")   
        return sol.y[0:n,:],sol.y[n:2*n,:],sol.y[-1,0],sol.x[:]    

    def solveOLIterativ(self,startState, endT,numbOfEval, ivpSolver='BDF'): # Solve open-loop optimal control problem iteratively     
        pres           = 10**(-6) 
        eps            = 10**(-6)
        maxIter        = 1000 
      
        def funcEval(control):
            def funState(t, x): return self.f(x) + self.g(x) @ control(t)    
            solState    = solve_ivp(funState, np.array([0,endT ]),  startState ,  method=ivpSolver, vectorized=False, max_step = np.inf,rtol = pres, atol = pres )
            state       = interp1d(solState.t,solState.y,kind = "cubic",fill_value = "extrapolate") 
            nonIntCosts = lambda t: (self.stateWeight*np.sum(( state(t))**2,0) + self.controlWeight*np.sum((control(t))**2,0))
            fe          = quad(nonIntCosts,0,endT)[0]
            return state, fe

        def cofuncEval(state,control):                      
            def funCoState(t, p): return -self.jacobi_of_f_transposed_dot_p(state(t),p)-2*state(t) * self.stateWeight
            pend         = state(endT)*0
            solCoState   = solve_ivp(funCoState, [endT,0], pend ,  method=ivpSolver, vectorized=False, max_step = np.inf,rtol = pres, atol = pres )
            coState      = interp1d(np.flip(solCoState.t), np.flip(solCoState.y,1),kind = "cubic",fill_value = "extrapolate")  
            grad         = lambda t: 2*self.controlWeight*control(t) + self.g(state(t)).T.dot(coState(t)) 
            gradNormFunc = lambda t: np.sum((grad(t))**2,0)
            normGrad     = np.sqrt(quad(gradNormFunc,0,endT)[0])
            dir          = lambda t: (-1)*grad(t)
            return dir,grad,normGrad,coState

        controlOld          = interp1d(np.linspace(0,endT,numbOfEval),np.zeros((self.g(startState).shape[1],numbOfEval)),kind = "cubic",fill_value = "extrapolate") 
        state, fe           = funcEval(controlOld)
        dirOld, grad, normGrad,coState = cofuncEval(state,controlOld)            
        print("iter "+ str(-2) +", fe =  " + str(fe) + ", normGrad = " + str(normGrad)) 
        control             = interp1d(np.linspace(0,endT,numbOfEval),controlOld(np.linspace(0,endT,numbOfEval)) - 1/(-normGrad) * dirOld(np.linspace(0,endT,numbOfEval)),kind = "cubic",fill_value = "extrapolate") 
        state, fe           = funcEval(control)            
        dir, grad, normGrad,coState = cofuncEval(state,control)
        print("iter "+ str(-1) +", fe =  " + str(fe) + ", normGrad = " + str(normGrad))
        
        j = 0
        while normGrad>eps and j<maxIter:
            forNormFunc1                = lambda t: np.sum(( (control(t)-controlOld(t)) * (dir(t)-dirOld(t))),0)
            forNormFunc2                = lambda t: np.sum( ((dir(t)-dirOld(t)) * (dir(t)-dirOld(t))) ,0 )
            alpha                       = quad(forNormFunc2,0,endT)[0]/quad(forNormFunc1,0,endT)[0]
            alpha                       = np.min([alpha,-alpha])
            controlOld                  = control
            dirOld                      = dir
            control                     = interp1d(np.linspace(0,endT,numbOfEval),control(np.linspace(0,endT,numbOfEval)) - 1/alpha * dir(np.linspace(0,endT,numbOfEval)),kind = "cubic",fill_value = "extrapolate") 
            state, fe                   = funcEval(control)            
            dir, grad, normGrad,coState = cofuncEval(state,control)
            j                           = j+1

            if normGrad<10:
               pres = 10**(-8)
            if normGrad<1:
               pres = 10**(-10)
            if normGrad<0.1:
               pres = 10**(-12)
            if normGrad<0.01:             
               pres    = 10**(-14)       
            if normGrad<0.0001:
               pres = 10**(-14)   
            print("iter "+ str(j-1) +", fe =  " + str(fe) + ", normGrad = " + str(normGrad)+ ", alpha = " + str(alpha))      

        return state(np.linspace(0,endT,numbOfEval)),coState(np.linspace(0,endT,numbOfEval)),fe ,np.linspace(0,endT,numbOfEval),


class HEDirichlet1D(Model): # Nonlinear Heat Equation with Dirichlet boundary conditions in 1D
    def __init__(self, NumbNodes, para): 
        self.NumbNodes     = NumbNodes
        self.alpha         = para[0]  
        self.beta          = para[1]  
        self.stateWeight   = para[2]  
        self.controlWeight = para[3]  
        self.gamma         = para[4]  
        self.controlDim    = 4

        self.makeFEMSystem(NumbNodes)

        self.matrixKGain    = la.solve_continuous_are( self.alpha * self.lapA, self.B, self.stateWeight*np.eye(self.A.shape[0]), self.controlWeight*np.eye(self.B.shape[1]), e=None, s=None, balanced=True)
        self.stableControl  = lambda x: (- (1/(self.controlWeight)) * self.g(x).T @ (self.matrixKGain @ x))*0
        self.lambdaMin      = np.linalg.eigvals(self.matrixKGain).min().real
        self.lambdaMax      = np.linalg.eigvals(self.matrixKGain).max().real
        self.trueVFKnown    = False
        self.trueVF         = lambda x:  np.sum(x * (self.matrixKGain  @ x),0)
        self.getMuFromSr    = lambda x,srX: (-0.5/self.controlWeight)* (self.g(x).T @ srX) 
        self.getF           = lambda x,muX: self.f(x)+self.g(x)@muX
        self.getRHS         = lambda x,muX: -self.stateWeight * np.sum(x**2,0) -self.controlWeight * np.sum(muX**2,0) 

    def f(self,x): return self.alpha * self.lapA@x + self.beta * self.invL.T@(self.invL@((self.A@x)**2 -(self.A@x)**3))
    def g(self,x): return self.B
    def jacobi_of_f_transposed_dot_p(self,x,p):
        Bp     = self.invL.T@(self.invL@p)
        return self.alpha * self.lapA.T @ p + 2* self.beta * self.A.T @ (Bp*(self.A@x)) - 3*self.beta * self.A.T @ (Bp*(self.A@x)**2)

    def plotSolution(self,solY): # Plot solution of the heat equation
            h      = 1/(self.NumbNodes+1)
            center = np.linspace(h,1-h,self.NumbNodes)
            def gauss(y,x):
                y    = np.atleast_2d(y).T
                x    = np.atleast_2d(x)
                gram = np.exp(-self.gamma*(x**2 +y**2 - 2 * y @ x)) * x * (1-x) * y * (1-y)
                return gram
            x        = np.linspace(0,1,1000)
            X        = gauss(x,center)
            fig      = plt.figure()
            ax       = fig.add_subplot(111)

            def update_currentPlot(num):
                ax.clear()
                ax.set_xlabel('X')
                ax.set_ylim(-1,1)
                ax.set_ylabel('Y')
                ax.set_xlim(0,1)
                return  ax.plot(x, X@(solY[:,num]))
            line_ani = animation.FuncAnimation(fig, update_currentPlot,solY.shape[1], interval=3,repeat=False)
            plt.show()

    def makeFEMSystem(self,NumbNodes):# Create FEM system for spatial discretization of the heat equation
        h      = 1/(NumbNodes+1)
        center = np.linspace(h,1-h,NumbNodes)

        def gauss(y,x):
            y    = np.atleast_2d(y).T
            x    = np.atleast_2d(x)
            gram = np.exp(-self.gamma*(x**2 +y**2 - 2 * y @ x)) * x * (1-x) * y * (1-y)
            return gram

        def lapGauss(y,x):
            y    = np.atleast_2d(y).T
            x    = np.atleast_2d(x)
            gram = np.exp(-self.gamma*(x**2 +y**2 - 2 * y @ x)) * x * (1-x) * (-4*self.gamma**2 * (y-1)*y*(x-y)**2+self.gamma*x*(4-8*y)+2*self.gamma*y*(5*y-3)-2)
            return gram

        self.A    = gauss(center,center)
        self.invL = np.linalg.solve(np.linalg.cholesky(self.A),np.eye(center.shape[0]))
        self.lapA = np.linalg.solve(self.A,lapGauss(center,center))
        self.B    = np.zeros((center.shape[0],4))

        def xi1(x): return 1*(x>=0.1) * (x<=0.2) 
        def xi2(x): return 1*(x>=0.3) * (x<=0.4) 
        def xi3(x): return 1*(x>=0.6) * (x<=0.7) 
        def xi4(x): return 1*(x>=0.8) * (x<=0.9) 

        for i in range(center.shape[0]):
            if xi1(center[i])==1:
                self.B[i,0] = 1
            if xi2(center[i])==1:
                self.B[i,1] = 1
            if xi3(center[i])==1:
                self.B[i,2] = 1
            if xi4(center[i])==1:
                self.B[i,3] = 1


class VanDerPol(Model): # Van der Pol oscillator
    def __init__(self,stateWeight,controlWeight):
        self.stateWeight   = stateWeight
        self.controlWeight = controlWeight
        self.matrixKGain   = la.solve_continuous_are(np.c_[np.r_[0,1],np.r_[1,1]],np.atleast_2d(np.array([0,1])).T,stateWeight* np.eye(2), controlWeight, e=None, s=None, balanced=True)
        self.lambdaMin     = np.linalg.eigvals(self.matrixKGain).min().real
        self.lambdaMax     = np.linalg.eigvals(self.matrixKGain).max().real
        self.stableControl = lambda x: np.atleast_2d(-(1/self.controlWeight) * np.sum(self.g(x) * (self.matrixKGain  @ x),0)) 
        self.getMuFromSr   = lambda x,srX: (-0.5/self.controlWeight)* (self.g(x).T @ srX) 
        self.getF          = lambda x,muX: self.f(x)+self.g(x)@muX
        self.getRHS        = lambda x,muX: -self.stateWeight * np.sum(x**2,0) -self.controlWeight * np.sum(muX**2,0) 
        self.controlDim    = 2

    def f(self,x):                              return np.array([x[1],-x[0]+x[1]*(1-x[0]**2)])
    def g(self,x):                              return np.array([[0,1]]).T
    def jacobi_of_f_transposed_dot_p(self,x,p): return np.array([-p[1]-2*p[1]*x[0]*x[1],p[0]+p[1]*(1-x[0]**2)]) 


class ToyExample: # Toy example from the paper
    def __init__(self):
        self.f             = lambda x: np.c_[-x[0,:]+x[1,:],-0.5*(x[0,:]+x[1,:])+0.5*x[1,:]*np.sin(x[0,:])**2].T
        self.g             = lambda x: np.c_[0*x[0,:],np.sin(x[0,:])].T
      
        self.stateWeight   = 1
        self.controlWeight = 1
        self.matrixKGain   = la.solve_continuous_are(np.c_[np.r_[-1,1],np.r_[-0.5,-0.5]],np.atleast_2d(np.array([0,1])).T,self.stateWeight* np.eye(2), self.controlWeight, e=None, s=None, balanced=True)
        self.lambdaMin     = np.linalg.eigvals(self.matrixKGain).min().real
        self.lambdaMax     = np.linalg.eigvals(self.matrixKGain).max().real
        self.stableControl = lambda x: (-3/2)*np.sin(x[0,:])*(x[0,:]+x[1,:])
        self.trueVF        = lambda x: 0.5*x[0,:]**2+x[1,:]**2
        self.getMuFromSr   = lambda x,srX: (-0.5/self.controlWeight)* np.sum((self.g(x) * srX),0) 
        self.getF          = lambda x,muX: self.f(x)+self.g(x)*muX
        self.getRHS        = lambda x,muX: -self.stateWeight * np.sum(x**2,0) -self.controlWeight * muX**2
        self.controlDim    = 1
      
    def f(self,x):                              return np.c_[-x[0,:]+x[1,:],-0.5*(x[0,:]+x[1,:])+0.5*x[1,:]*np.sin(x[0,:])**2].T
    def g(self,x):                              return np.c_[0*x[0,:],np.sin(x[0,:])].T
    def jacobi_of_f_transposed_dot_p(self,x,p): pass

