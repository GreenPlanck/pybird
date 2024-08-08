#from pybird.module import *
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
# from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController
# import jax
# from jax.numpy import exp,log,linspace
# jax.config.update("jax_enable_x64", True)
from scipy.integrate import odeint
from numpy import exp,log,linspace,array


# numerical D,DD+,D-,DD-
# note that D and f are function of x while G Y are functio of a
class GreenFunction(object):

    def __init__(self, Omega0_m, w=None,wa=0, quintessence=False, Omega0_k=0., vectorize=False):
        self.vectorize = vectorize
        self.Omega0_m = Omega0_m
        if self.vectorize:
            self.Omega0_m = np.array(Omega0_m)
            self.Omega0_k = np.array(Omega0_k)
            self.w = np.array(w) if w is not None else None
        self.OmegaL_by_Omega_m = (1.-self.Omega0_m-Omega0_k)/self.Omega0_m
        
        self.x0 = -12. # the initial time for ODE
        self.lo = exp(self.x0 )  # the lowest integration bound of scale factor x, x=lna

        self.w0 = w
        self.wa = wa
        self.quintessence = quintessence

        self.epsrel = 1e-4
        self.H0 =1
        self.interpD()

    def w(self,x):
        a = exp(x)
        return self.w0+self.wa*(1-a)
    
    def dwdx(self,x):
        a = exp(x)
        return -self.wa*a
    
    def hw(self,x):
        a = exp(x)
        return (self.wa-a*self.wa+(self.w0+self.wa)*x)/(x+1e-20)
    
    def H(self,x):
        a = exp(x)
        return self.H0*(self.Omega0_m*a**(-3)+(1.-self.Omega0_m)*a**(-3-3*self.hw(x)))**0.5
    
    def dHdx(self,x):
        return -1.5*self.H(x)*self.Om(x)-1.5*(1+self.w(x))*self.H(x)*self.Ode(x)
    
    def d2Hdx2(self,x):
        return -1.5*self.dHdx(x)*self.Om(x)-1.5*self.H(x)*self.dOmdx(x)-1.5*self.dwdx(x)*self.H(x)*self.Ode(x)-1.5*(1+self.w(x))*self.dHdx(x)*self.Ode(x)-1.5*(1+self.w(x))*self.H(x)*self.dOdedx(x)
        # return -1.5*(self.dOmdx(x)*self.C(x)*self.H(x)
        #              +self.Om(x)*self.dCdx(x)*self.H(x)
        #              +self.Om(x)*self.C(x)*self.dHdx(x))
    def Om(self,x):
        a = exp(x)
        return self.Omega0_m*a**-3/(self.H(x)**2/self.H0**2)
    
    def Ode(self,x):
        a = exp(x)
        return (1.-self.Omega0_m)*a**(-3.-3.*self.hw(x))/(self.H(x)**2/self.H0**2)
    
    def dOmdx(self,x):
        #return 3*self.Om(x)*self.Ode(x)*(1+self.w(x))+3*self.Om(x)*(self.Om(x)-1)
        return 3*self.w(x)*self.Om(x)*self.Ode(x)
    
    def dOdedx(self,x):
        #return 3*self.Ode(x)*self.Om(x)+3*self.Ode(x)*(1+self.w(x))*(self.Ode(x)-1)
        return -3*self.w(x)*self.Om(x)*self.Ode(x)
    
    def C(self, x):
        if self.quintessence: return 1. + (1.+self.w(x)) * self.Ode(x)/self.Om(x)
        else: return 1.

    def dCdx(self,x):
        if self.quintessence: return self.Ode(x)/self.Om(x)*(-3*self.w(x)*(1+self.w(x))+self.dwdx(x))
        else: return 0.
    
    def get_ini(self,xi):
        ai = exp(xi)
        Di = ai
        dDi = ai
        Dminusi = ai**(-3/2)
        dDminusi = -3./2.*ai**(-3./2.)
        return dDi,Di,dDminusi,Dminusi  
    


    def vector_field(self,y,x):
        a = exp(x)
        dD,D, dDminus,Dminus= y
        
        epsilon = - self.dHdx(x)/self.H(x)+self.dCdx(x)/self.C(x)
        #epsilon = 2-0.5*(1-3*self.w(x)*self.Ode(x))+self.dCdx(x)/self.C(x)
        
        F = 1.5*self.Om(x) *self.C(x)#self.dHdx(x)**2/self.H(x)**2+self.d2Hdx2(x)/self.H(x)+2*self.dHdx(x)/self.H(x)-self.dCdx(x)/self.C(x)*self.dHdx(x)/self.H(x)
        
        #F = -self.dHdx(x)/self.H(x) 
        # this eqaution only valid for wcdm-cq not wcdm or w0wa
        #F = self.dHdx(x)**2/self.H(x)**2+self.d2Hdx2(x)/self.H(x)+2*self.dHdx(x)/self.H(x)-self.dCdx(x)/self.C(x)*self.dHdx(x)/self.H(x)


        D_D = dD
        D_dD = (epsilon-2)*dD+F*D

        D_Dminus = dDminus
        D_dDminus = (epsilon-2)*dDminus+F*Dminus

        
        D_y = [D_dD,D_D,D_dDminus,D_Dminus]
        return D_y  
    
    def interpD(self):
        
        x0 = self.x0
        x1 = 0.
        x = linspace(x0, x1, 500)
        y0 = array((self.get_ini(x0)))
        sol = odeint(self.vector_field, y0, x).T

        self.Darr = sol[1]
        self.dDarr = sol[0]/exp(x)  #dDda
        self.Dminusarr = sol[3]
        self.dDminusarr = sol[2]/exp(x)
        
        self.D = CubicSpline(x,self.Darr)
        self.DD = CubicSpline(x,self.dDarr)  #dDda(x)
        self.Dminus = CubicSpline(x,self.Dminusarr)
        self.DDminus = CubicSpline(x,self.dDminusarr)

    # def vector_field(self,x, y,args):
    #     a = exp(x)
    #     dD,D, dDminus,Dminus= y
        
    #     epsilon = - self.dHdx(x)/self.H(x)+self.dCdx(x)/self.C(x)
    #     #epsilon = 2-0.5*(1-3*self.w(x)*self.Ode(x))+self.dCdx(x)/self.C(x)
        
    #     F = 1.5*self.Om(x) *self.C(x)#self.dHdx(x)**2/self.H(x)**2+self.d2Hdx2(x)/self.H(x)+2*self.dHdx(x)/self.H(x)-self.dCdx(x)/self.C(x)*self.dHdx(x)/self.H(x)
        
    #     #F = -self.dHdx(x)/self.H(x) 
    #     # this eqaution only valid for wcdm-cq not wcdm or w0wa
    #     #F = self.dHdx(x)**2/self.H(x)**2+self.d2Hdx2(x)/self.H(x)+2*self.dHdx(x)/self.H(x)-self.dCdx(x)/self.C(x)*self.dHdx(x)/self.H(x)


    #     D_D = dD
    #     D_dD = (epsilon-2)*dD+F*D

    #     D_Dminus = dDminus
    #     D_dDminus = (epsilon-2)*dDminus+F*Dminus

        
    #     D_y = D_dD,D_D,D_dDminus,D_Dminus
    #     return D_y  
    
    # def interpD(self):
        
    #     stepsize_controller = PIDController(rtol=1e-8, atol=1e-10)#ConstantStepSize()#
    #     term = ODETerm(self.vector_field)
    #     solver = Dopri5()
    #     x0 = self.x0
    #     x1 = 0.
    #     dt0 = None
    #     y0 = self.get_ini(x0)
    #     saveat = SaveAt(ts=linspace(x0, x1, 500))
    #     sol = diffeqsolve(term, solver, x0, x1, dt0, y0, saveat=saveat,stepsize_controller=stepsize_controller)
        
    #     x = sol.ts
    #     #a = exp(x)

    #     self.Darr = sol.ys[1]
    #     self.dDarr = sol.ys[0]/exp(x)  #dDda
    #     self.Dminusarr = sol.ys[3]
    #     self.dDminusarr = sol.ys[2]/exp(x)
        
    #     self.D = CubicSpline(x,self.Darr)
    #     self.DD = CubicSpline(x,self.dDarr)  #dDda(x)
    #     self.Dminus = CubicSpline(x,self.Dminusarr)
    #     self.DDminus = CubicSpline(x,self.dDminusarr)
        

    def fplus(self, x):
        """Growth rate"""
        return exp(x)*self.DD(x) / self.D(x)

    def fminus(self, x):
        """Decay rate"""
        return exp(x)*self.DDminus(x) / self.Dminus(x)

    def W(self,x):
        """Wronskian"""
        return (self.DDminus(x) * self.D(x) - self.DD(x) * self.Dminus(x))

 #greens functions
    def G1d(self, a, ai):
        x = log(a)
        xi = log(ai)
        return(self.DDminus(xi)*self.D(x)-self.DD(xi)*self.Dminus(x))/(ai*self.W(xi))
    def G2d(self, a, ai):
        x = log(a)
        xi = log(ai)
        return self.fplus(xi)*(self.Dminus(x)*self.D(xi)-self.D(x)*self.Dminus(xi))/(ai*ai*self.W(xi))
    def G1t(self, a, ai):
        x = log(a)
        xi = log(ai)
        return a*(self.DDminus(xi)*self.DD(x)-self.DD(xi)*self.DDminus(x))/(self.fplus(x)*ai*self.W(xi))
    def G2t(self, a, ai):
        x = log(a)
        xi = log(ai)
        return a*self.fplus(xi)*(self.DDminus(x)*self.D(xi)-self.DD(x)*self.Dminus(xi))/(self.fplus(x)*ai*ai*self.W(xi))

    # second order coefficients
    def I1d(self, ai, a):
        x = log(a)
        xi = log(ai)
        return self.fplus(xi)*self.D(xi)**2*self.G1d(a,ai)/self.D(x)**2 / self.C(xi)
    def I2d(self, ai, a):
        x = log(a)
        xi = log(ai)
        return self.fplus(xi)*self.D(xi)**2*self.G2d(a,ai)/self.D(x)**2 / self.C(xi)
    def I1t(self, ai, a):
        x = log(a)
        xi = log(ai)
        return self.fplus(xi)*self.D(xi)**2*self.G1t(a,ai)/self.D(x)**2 / self.C(xi)
    def I2t(self, ai, a):
        x = log(a)
        xi = log(ai)
        return self.fplus(xi)*self.D(xi)**2*self.G2t(a,ai)/self.D(x)**2 / self.C(xi)

    # second order time integrals
    def mG1d(self, a):
        if self.vectorize:
            return quad_vec(self.I1d,self.lo,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
        else:
            return quad(self.I1d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mG2d(self, a):
        if self.vectorize:
            return quad_vec(self.I2d,self.lo,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
        else:
            return quad(self.I2d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mG1t(self, a):
        if self.vectorize:
            return quad_vec(self.I1t,self.lo,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
        else:
            return quad(self.I1t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mG2t(self, a):
        if self.vectorize:
            return quad_vec(self.I2t,self.lo,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
        else:
            return quad(self.I2t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]

    # quintessence time function
    def G(self, a):
        return self.mG1d(a) + self.mG2d(a)

    # third order coefficients
    def IU1d(self, ai, a):
        x = log(a)
        xi = log(ai)
        return self.fplus(xi)*self.mG1d(ai)*self.G1d(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
    def IU2d(self, ai, a):
        x = log(a)
        xi = log(ai)
        return self.fplus(xi)*self.mG2d(ai)*self.G1d(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
    def IU1t(self, ai, a):
        x = log(a)
        xi = log(ai)
        return self.fplus(xi)*self.mG1d(ai)*self.G1t(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
    def IU2t(self, ai, a):
        x = log(a)
        xi = log(ai)
        return self.fplus(xi)*self.mG2d(ai)*self.G1t(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)

    def IV11d(self, ai, a):
        x = log(a)
        xi = log(ai)
        return self.fplus(xi)*self.mG1t(ai)*self.G1d(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
    def IV12d(self, ai, a):
        x = log(a)
        xi = log(ai)
        return self.fplus(xi)*self.mG1t(ai)*self.G2d(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
    def IV21d(self, ai, a):
        x = log(a)
        xi = log(ai)
        return self.fplus(xi)*self.mG2t(ai)*self.G1d(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
    def IV22d(self, ai, a):
        x = log(a)
        xi = log(ai)
        return self.fplus(xi)*self.mG2t(ai)*self.G2d(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)

    def IV11t(self, ai,a):
        x = log(a)
        xi = log(ai)
        return self.fplus(xi)*self.mG1t(ai)*self.G1t(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
    def IV12t(self, ai,a):
        x = log(a)
        xi = log(ai)
        return self.fplus(xi)*self.mG1t(ai)*self.G2t(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
    def IV21t(self, ai,a):
        x = log(a)
        xi = log(ai)
        return self.fplus(xi)*self.mG2t(ai)*self.G1t(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
    def IV22t(self, ai,a):
        x = log(a)
        xi = log(ai)
        return self.fplus(xi)*self.mG2t(ai)*self.G2t(a,ai)*(self.D(xi)/self.D(x))**3 / self.C(xi)
   
    # third order time integrals
    def mU1d(self, a):
        if self.vectorize:
            return quad_vec(self.IU1d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]

        else:
            return quad(self.IU1d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mU2d(self, a):
        if self.vectorize:
            return quad_vec(self.IU2d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IU2d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mU1t(self, a):
        if self.vectorize:
            return quad_vec(self.IU1t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IU1t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mU2t(self, a):
        if self.vectorize:
            return quad_vec(self.IU2t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IU2t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]

    def mV11d(self, a):
        if self.vectorize:
            return quad_vec(self.IV11d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV11d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mV12d(self, a):
        if self.vectorize:
            return quad_vec(self.IV12d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV12d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mV21d(self, a):
        if self.vectorize:
            return quad_vec(self.IV21d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV21d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mV22d(self, a):
        if self.vectorize:
            return quad_vec(self.IV22d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV22d,self.lo,a,args=(a,), epsrel=self.epsrel)[0]

    def mV11t(self, a):
        if self.vectorize:
            return quad_vec(self.IV11t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV11t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mV12t(self, a):
        if self.vectorize:
            return quad_vec(self.IV12t,self.lo,a,args=(a,), epsrel=self.epsrel, epsabs=1.49e-5)[0]
        else:
            return quad(self.IV12t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mV21t(self, a):
        if self.vectorize:
            return quad_vec(self.IV21t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV21t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
    def mV22t(self, a):
        if self.vectorize:
            return quad_vec(self.IV22t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]
        else:
            return quad(self.IV22t,self.lo,a,args=(a,), epsrel=self.epsrel)[0]

    def Y(self, a):
        if self.quintessence: return -3/14.*self.G(a)**2 + self.mV11d(a) + self.mV12d(a)
        else: return -3/14. + self.mV11d(a) + self.mV12d(a)


# ===========================================

    # #greens functions
    # def G1d(self, x, xi):
    #     ai = exp(xi)
    #     return(self.DDminus(xi)*self.D(x)-self.DD(xi)*self.Dminus(x))/(ai*self.W(xi))
    # def G2d(self, x, xi):
    #     ai = exp(xi)
    #     return self.fplus(xi)*(self.Dminus(x)*self.D(xi)-self.D(x)*self.Dminus(xi))/(ai*ai*self.W(xi))
    # def G1t(self, x, xi):
    #     a = exp(x)
    #     ai = exp(xi)
    #     return a*(self.DDminus(xi)*self.DD(x)-self.DD(xi)*self.DDminus(x))/(self.fplus(x)*ai*self.W(xi))
    # def G2t(self, x, xi):
    #     a = exp(x)
    #     ai = exp(xi)
    #     return a*self.fplus(xi)*(self.DDminus(x)*self.D(xi)-self.DD(x)*self.Dminus(xi))/(self.fplus(x)*ai*ai*self.W(xi))

    # # second order coefficients
    # # the last exp(x) comes from the change of integral variable
    # def I1d(self, xi, x):
    #     return self.fplus(xi)*self.D(xi)**2*self.G1d(x,xi)/self.D(x)**2 / self.C(xi)*exp(xi)
    # def I2d(self, xi, x):
    #     return self.fplus(xi)*self.D(xi)**2*self.G2d(x,xi)/self.D(x)**2 / self.C(xi)*exp(xi)
    # def I1t(self, xi, x):
    #     return self.fplus(xi)*self.D(xi)**2*self.G1t(x,xi)/self.D(x)**2 / self.C(xi)*exp(xi)
    # def I2t(self, xi, x):
    #     return self.fplus(xi)*self.D(xi)**2*self.G2t(x,xi)/self.D(x)**2 / self.C(xi)*exp(xi)

    # # second order time integrals
    # def mG1d(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.I1d,lo,x,args=(x,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
    #     else:
    #         return quad(self.I1d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mG2d(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.I2d,lo,x,args=(x,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
    #     else:
    #         return quad(self.I2d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mG1t(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.I1t,lo,x,args=(x,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
    #     else:
    #         return quad(self.I1t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mG2t(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.I2t,lo,x,args=(x,), epsrel=self.epsrel, epsabs=1.49e-08)[0]
    #     else:
    #         return quad(self.I2t,lo,x,args=(x,), epsrel=self.epsrel)[0]

    # # quintessence time function
    # def G(self, x):
    #     return self.mG1d(x) + self.mG2d(x)

    # # third order coefficients
    # def IU1d(self, xi, x):
    #     return self.fplus(xi)*self.mG1d(xi)*self.G1d(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
    # def IU2d(self, xi, x):
    #     return self.fplus(xi)*self.mG2d(xi)*self.G1d(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
    # def IU1t(self, xi, x):
    #     return self.fplus(xi)*self.mG1d(xi)*self.G1t(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
    # def IU2t(self, xi, x):
    #     return self.fplus(xi)*self.mG2d(xi)*self.G1t(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)

    # def IV11d(self, xi, x):
    #     return self.fplus(xi)*self.mG1t(xi)*self.G1d(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
    # def IV12d(self, xi, x):
    #     return self.fplus(xi)*self.mG1t(xi)*self.G2d(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
    # def IV21d(self, xi, x):
    #     return self.fplus(xi)*self.mG2t(xi)*self.G1d(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
    # def IV22d(self, xi, x):
    #     return self.fplus(xi)*self.mG2t(xi)*self.G2d(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)

    # def IV11t(self, ai,a):
    #     return self.fplus(xi)*self.mG1t(xi)*self.G1t(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
    # def IV12t(self, ai,a):
    #     return self.fplus(xi)*self.mG1t(xi)*self.G2t(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
    # def IV21t(self, ai,a):
    #     return self.fplus(xi)*self.mG2t(xi)*self.G1t(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
    # def IV22t(self, ai,a):
    #     return self.fplus(xi)*self.mG2t(xi)*self.G2t(x,xi)*(self.D(xi)/self.D(x))**3 / self.C(xi)*exp(xi)
   
    # # third order time integrals
    # def mU1d(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IU1d,lo,x,args=(x,), epsrel=self.epsrel)[0]

    #     else:
    #         return quad(self.IU1d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mU2d(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IU2d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IU2d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mU1t(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IU1t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IU1t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mU2t(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IU2t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IU2t,lo,x,args=(x,), epsrel=self.epsrel)[0]

    # def mV11d(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IV11d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IV11d,lo,x,args=(a,), epsrel=self.epsrel)[0]
    # def mV12d(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IV12d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IV12d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mV21d(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IV21d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IV21d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mV22d(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IV22d,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IV22d,lo,x,args=(x,), epsrel=self.epsrel)[0]

    # def mV11t(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IV11t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IV11t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mV12t(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IV12t,lo,x,args=(x,), epsrel=self.epsrel, epsabs=1.49e-5)[0]
    #     else:
    #         return quad(self.IV12t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mV21t(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IV21t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IV21t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    # def mV22t(self, x):
    #     if self.vectorize:
    #         return quad_vec(self.IV22t,lo,x,args=(x,), epsrel=self.epsrel)[0]
    #     else:
    #         return quad(self.IV22t,lo,x,args=(x,), epsrel=self.epsrel)[0]

    # def Y(self, x):
    #     if self.quintessence: return -3/14.*self.G(x)**2 + self.mV11d(x) + self.mV12d(x)
    #     else: return -3/14. + self.mV11d(x) + self.mV12d(x)