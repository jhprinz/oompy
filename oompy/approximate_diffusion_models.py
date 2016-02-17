
# coding: utf-8

# Construct a diffusion model dx=-V'(x)dt+sqrt(2/beta)*dW
# The stationary distribution is exp(-beta * V(x))

# In[1]:

import numpy as np
import scipy.linalg as linalg

import numba
from numba import jit,int_,double

from math import exp


# In[2]:

@jit(numba.typeof(np.array([1]))(int_,double[:,:],double[:]),nopython=True)
def _Markov_chain_generate(N,P,start_prob):
    n=P.shape[0]
    P_cumsum=np.empty((n,n))
    for i in range(n):
        P_cumsum[i,0]=P[i,0]
        for j in range(1,n):
            P_cumsum[i,j]=P_cumsum[i,j-1]+P[i,j]
    start_cumsum=np.empty(n)
    start_cumsum[0]=start_prob[0]
    for i in range(1,n):
        start_cumsum[i]=start_cumsum[i-1]+start_prob[i]
    traj=np.empty(N,dtype=np.int_)
    for t in range(N):
        u=np.random.rand()
        if t==0:
            for s in range(n):
                if u<=start_cumsum[s]:
                    break
        else:
            for s in range(n):
                if u<=P_cumsum[traj[t-1],s]:
                    break
        traj[t]=s
    return traj


# In[3]:

# C=A_for_corr*expm(Q^lag)*B_for_corr for lag>0
# C=C_for_corr for lag=0
def _correlation(A_for_corr,B_for_corr,C_for_corr,Q,lags):
    actual_lags=np.maximum(lags,0)
    if np.isscalar(actual_lags):
        if actual_lags==0:
            return C_for_corr,actual_lags
        C=A_for_corr.dot(linalg.expm(Q*actual_lags)).dot(B_for_corr)
        C=0.5*(C+C.T)
        return C,actual_lags
    else:
        C_mem=np.empty(actual_lags.shape+C_for_corr.shape)
        ad=0
        AD=A_for_corr.copy()
        d=0
        D=np.identity(Q.shape[0])
        for n in range(actual_lags.shape[0]):
            i=actual_lags[n]
            if i-ad==d:
                AD=AD.dot(D)
                ad=i
                C=AD.dot(B_for_corr)
            elif i-ad>=0:
                d=i-ad
                D=linalg.expm(Q*d)
                AD=AD.dot(D)
                ad=i
                C=AD.dot(B_for_corr)
            else:
                ad=i
                AD=A_for_corr.dot(linalg.expm(Q*ad))
                d=0
                D=np.identity(Q.shape[0])
                C=AD.dot(B_for_corr)
            if i==0:
                C=C_for_corr.copy()
            C_mem[n]=0.5*(C+C.T)
        return C_mem,actual_lags


# In[7]:

class OneDimensionalModel:

    def __init__(self,V,beta,lb,ub,grid_num=100,dt=1.0):
        """A one-dimensional diffusion model with a given potential function and temperature
        Parameters
        ----------
        V : function
            potential function
        beta : double
            the temperature parameter. The stationary distribution of the model is given by exp(-beta*V)
        lb,ub : double
            the lower and upper bounds of the state space
        grid_num : int, optional, default=100
            the number of discrete spatial grids for approximation
        dt : double, optional, default=1.
            the simulation step size
        """
        self.potential_function=V
        self.beta=beta+0.0
        self.step_size=dt+0.0
        lb=lb+0.
        ub=ub+0.
        tmp_center_list=np.linspace(lb,ub,num=grid_num+1,endpoint=True)
        self.center_list=0.5*(tmp_center_list[1:]+tmp_center_list[:-1])
        h=(ub-lb)/grid_num
        self.width=h
        delta=beta*h*h
        self.Q=np.zeros((grid_num,grid_num))
        self.pii=np.zeros(grid_num)
        for i in range(grid_num):
            Vi=V(self.center_list[i])
            tmp_sum=0.0
            for j in [i-1,i+1]:
                if j>=0 and j<grid_num:
                    Vij=V(0.5*(self.center_list[i]+self.center_list[j]))
                    self.Q[i,j]=exp(-beta*(Vij-Vi))/delta
                    tmp_sum+=self.Q[i,j]
            self.Q[i,i]=-tmp_sum
            self.pii[i]=exp(-beta*Vi)
        self.pii=self.pii/self.pii.sum()
        self.its=-1.0/np.sort(np.real(linalg.eigvals(self.Q)))[1:][::-1]
        self.its=np.concatenate((np.array([np.inf]),self.its))
        self.P=linalg.expm(self.Q*self.step_size)

    def simulate(self,step_num,x_0=None):
        """ Perform simulations
        Parameters
        ----------
        step_num : int
            number of simulation steps
        x_0 : double, optional, default=None
            the starting point for the simulation. If x_0 is None, the starting point is picked up
            according to the stationary distribution.

        Returns
        -------
        traj : ndarray(step_num,)
            a simulation trajectory
        """
        if x_0 is None:
            pii_0=self.pii
        else:
            pii_0=np.zeros(self.center_list.shape[0])
            pii_0[np.argmin(np.fabs(self.center_list-x_0))]=1.
        dtraj=_Markov_chain_generate(step_num,self.P,pii_0)
        traj=self.center_list[dtraj]+(np.random.rand(step_num)-0.5)*self.width
        if x_0 is not None:
            traj[0]=x_0
        return traj.reshape(-1,1)

    def simulate_with_noise(self,step_num,Sigma2,x_0=None):
        """ Perform simulations with Gaussian noise
        Parameters
        ----------
        step_num : int
            number of simulation steps
        Sigma2 : double
            variance of noise
        x_0 : double, optional, default=None
            the starting point for the simulation. If x_0 is None, the starting point is picked up
            according to the stationary distribution.

        Returns
        -------
        traj : ndarray(step_num,)
            a simulation trajectory
        """
        return self.simulate(step_num,x_0)+np.random.randn(step_num,1)*np.sqrt(Sigma2)

    def expectation(self):
        """ Compute the mean value of {x_t}
        Returns
        -------
        u : double
            mean value of {x_t}
        """
        return np.inner(self.center_list,self.pii)

    def correlation(self,lags):
        """ Compute the corrleation E[x_t*x_(t+tau)] at different lag times
        Parameters
        ----------
        lags : ndarray(k) or int list
            lag times for computing the correlation. (The unit of lag times is the simulation step size, i.e.,
            the lag time in unit of second = lags * self.step_size)

        Returns
        -------
        C_mem : ndarray(k)
            correlations
        actual_lags : ndarray(k)
            the actual lag times for computing the correlations. Note it may be different with lags
            if lags contains negative values.
        """
        A_for_corr=(self.center_list*self.pii).reshape(1,-1)
        B_for_corr=self.center_list.reshape(-1,1)
        C_for_corr=A_for_corr.dot(B_for_corr)+self.width*self.width/12.0
        tmp_C_mem,actual_lags=_correlation(A_for_corr,B_for_corr,C_for_corr,self.Q,np.array(lags)*self.step_size)
        actual_lags=actual_lags/self.step_size
        if np.isscalar(actual_lags):
            return tmp_C_mem[0,0],actual_lags
        else:
            C_mem=np.zeros(tmp_C_mem.shape[0])
            for i in range(tmp_C_mem.shape[0]):
                C_mem[i]=tmp_C_mem[i][0,0]
            return C_mem,actual_lags

    def correlation_with_noise(self,lags,Sigma2):
        """ Compute the corrleation E[x_t*x_(t+tau)] with noise at different lag times
        The results are the same with function correlation except for lag 0
        """
        A_for_corr=(self.center_list*self.pii).reshape(1,-1)
        B_for_corr=self.center_list.reshape(-1,1)
        C_for_corr=A_for_corr.dot(B_for_corr)+self.width*self.width/12.0+Sigma2
        tmp_C_mem,actual_lags=_correlation(A_for_corr,B_for_corr,C_for_corr,self.Q,np.array(lags)*self.step_size)
        actual_lags=actual_lags/self.step_size
        if np.isscalar(actual_lags):
            return tmp_C_mem[0,0],actual_lags
        else:
            C_mem=np.zeros(tmp_C_mem.shape[0])
            for i in range(tmp_C_mem.shape[0]):
                C_mem[i]=tmp_C_mem[i][0,0]
            return C_mem,actual_lags


# In[5]:

class TwoDimensionalModel:
    def __init__(self,V,beta,lb,ub,grid_num=np.array([100,100]),dt=1.0):
        """A two-dimensional diffusion model with a given potential function and temperature
        Parameters
        ----------
        V : function
            potential function
        beta : double
            the temperature parameter. The stationary distribution of the model is given by exp(-beta*V)
        lb,ub : ndarray(2)
            the lower and upper bounds at x-axis and y-axis of the state space
        grid_num : int, optional, default=100
            the number of discrete spatial grids along x-axis and y-axis.
            The total number of grid numbers is grid_num[0]*grid_num[1]
        dt : double, optional, default=1.
            the simulation step size
        """
        self.potential_function=V
        self.beta=beta+0.0
        self.step_size=dt+0.0
        self.state_num=grid_num[0]*grid_num[1]
        tmp_x=np.linspace(lb[0],ub[0],num=grid_num[0]+1,endpoint=True)
        tmp_x=0.5*(tmp_x[1:]+tmp_x[:-1])
        tmp_y=np.linspace(lb[1],ub[1],num=grid_num[1]+1,endpoint=True)
        tmp_y=0.5*(tmp_y[1:]+tmp_y[:-1])
        tmp_xx,tmp_yy=np.meshgrid(tmp_x,tmp_y)
        tmp_ii=np.array(range(self.state_num)).reshape(tmp_xx.shape)
        self.center_list=np.hstack((tmp_xx.reshape(-1,1),tmp_yy.reshape(-1,1)))
        h=(ub-lb)/grid_num
        delta=beta*h*h
        self.width=h
        self.Q=np.zeros((self.state_num,self.state_num))
        self.pii=np.zeros(self.state_num)
        move_step_mem=np.array([[-1,0],[1,0],[0,-1],[0,1]])
        for i in range(grid_num[1]):
            for j in range(grid_num[0]):
                k=tmp_ii[i,j]
                Vk=V(self.center_list[k])
                self.pii[k]=exp(-beta*Vk)
                for move_step_ind in range(4):
                    move_step=move_step_mem[move_step_ind]
                    i_new=i+move_step[0]
                    j_new=j+move_step[1]
                    if i_new>=0 and i_new<grid_num[1] and j_new>=0 and j_new<grid_num[0]:
                        k_new=tmp_ii[i_new,j_new]
                        V_new=V(0.5*(self.center_list[k]+self.center_list[k_new]))
                        if i==i_new:
                            #y coordinate is the same, x coordinate is different
                            self.Q[k,k_new]=exp(-beta*(V_new-Vk))/delta[0]
                        else:
                            #x coordinate is the same, y coordinate is different
                            self.Q[k,k_new]=exp(-beta*(V_new-Vk))/delta[1]
        self.Q[np.diag_indices(self.Q.shape[0])]=-self.Q.sum(axis=1)
        self.pii=self.pii/self.pii.sum()
        self.its=-1.0/np.sort(np.real(linalg.eigvals(self.Q)))[1:][::-1]
        self.its=np.concatenate((np.array([np.inf]),self.its))
        self.P=linalg.expm(self.Q*self.step_size)

    def simulate(self,step_num,x_0=None):
        """ Perform simulations
        Parameters
        ----------
        step_num : int
            number of simulation steps
        x_0 : ndarray(2), optional, default=None
            the starting point for the simulation. If x_0 is None, the starting point is picked up
            according to the stationary distribution.

        Returns
        -------
        traj : ndarray(step_num,2)
            a simulation trajectory
        """
        if x_0 is None:
            pii_0=self.pii
        else:
            pii_0=np.zeros(self.center_list.shape[0])
            pii_0[np.argmin((self.center_list[:,0]-x_0[0])**2+(self.center_list[:,1]-x_0[1])**2)]=1.
        dtraj=_Markov_chain_generate(step_num,self.P,pii_0)
        traj=self.center_list[dtraj]+(np.random.rand(step_num,2)-0.5)*self.width
        if x_0 is not None:
            traj[0]=x_0
        return traj

    def simulate_with_noise(self,step_num,Sigma2,x_0=None):
        """ Perform simulations with Gaussian noise
        Parameters
        ----------
        step_num : int
            number of simulation steps
        Sigma2 : ndarray(2,2)
            covariance matrix of noise
        x_0 : double, optional, default=None
            the starting point for the simulation. If x_0 is None, the starting point is picked up
            according to the stationary distribution.

        Returns
        -------
        traj : ndarray(step_num,2)
            a simulation trajectory
        """
        return self.simulate(step_num,x_0)+np.random.multivariate_normal(np.zeros(2),Sigma2,step_num)

    def expectation(self):
        """ Compute the mean value of {x_t}
        Returns
        -------
        u : ndarray(2)
            mean value of {x_t}
        """
        return self.pii.reshape(1,-1).dot(self.center_list)[0]

    def correlation(self,lags):
        """ Compute the corrleation E[x_t*x_(t+tau)] at different lag times
        Parameters
        ----------
        lags : ndarray(k) or int list
            lag times for computing the correlation. (The unit of lag times is the simulation step size, i.e.,
            the lag time in unit of second = lags * self.step_size)

        Returns
        -------
        C_mem : ndarray(k,2,2)
            C_mem[i] is the correlation matrix at lag time lags[i]
        actual_lags : ndarray(k)
            the actual lag times for computing the correlations. Note it may be different with lags
            if lags contains negative values.
        """
        A_for_corr=self.center_list.T*self.pii
        B_for_corr=self.center_list
        C_for_corr=A_for_corr.dot(B_for_corr)+np.diag(self.width*self.width/12.0)
        tmp_C_mem,actual_lags=_correlation(A_for_corr,B_for_corr,C_for_corr,self.Q,np.array(lags)*self.step_size)
        actual_lags=actual_lags/self.step_size
        return tmp_C_mem,actual_lags

    def correlation_with_noise(self,lags,Sigma2):
        """ Compute the corrleation E[x_t*x_(t+tau)] with noise at different lag times
        The results are the same with function correlation except for lag 0
        """
        A_for_corr=self.center_list.T*self.pii
        B_for_corr=self.center_list
        C_for_corr=A_for_corr.dot(B_for_corr)+np.diag(self.width*self.width/12.0)+Sigma2
        tmp_C_mem,actual_lags=_correlation(A_for_corr,B_for_corr,C_for_corr,self.Q,np.array(lags)*self.step_size)
        actual_lags=actual_lags/self.step_size
        return tmp_C_mem,actual_lags

