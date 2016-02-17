
# coding: utf-8

# In[4]:

import numpy as np
import scipy.linalg as linalg
import numba

from numba import jit,int_,double


# In[5]:

@jit(double[:,:,:](int_[:],double[:,:],numba.typeof([1])),nopython=True)
def _empirical_correlation_scan(actual_lags,eval_mem,traj_no_mem):
    T,eval_dim=eval_mem.shape
    lag_num=actual_lags.shape[0]
    C_mem=np.zeros((lag_num,eval_dim,eval_dim))
    for k in range(lag_num):
        lag=actual_lags[k]
        N=0.
        for t in range(T-lag):
            if traj_no_mem[t]==traj_no_mem[t+lag]:
                N+=1
                for i in range(eval_dim):
                    for j in range(eval_dim):
                         C_mem[k,i,j]+=eval_mem[t,i]*eval_mem[t+lag,j]
        for i in range(eval_dim):
            for j in range(i+1):
                C_mem[k,i,j]=0.5*(C_mem[k,i,j]+C_mem[k,j,i])/N
        for i in range(eval_dim):
            for j in range(i+1,eval_dim):
                C_mem[k,i,j]=C_mem[k,j,i]
    return C_mem


# In[6]:

class EmpiricalEstimator:

    def __init__(self,eval_trajs):
        """An empirical estimator for mean and correlations by simple average
        Parameters
        ----------
        eval_trajs : ndarray(T,d) or list of ndarray(T,d)
            trajectories of quantities needed to be estimated
        """
        if type(eval_trajs) is list:
            self.traj_no_mem=[]
            for i in range(len(eval_trajs)):
                self.traj_no_mem+=[i]*eval_trajs[i].shape[0]
        else:
            self.traj_no_mem=[0]*eval_trajs.shape[0]
        self.eval_mem=np.vstack(eval_trajs)

    def expectation(self):
        """ Compute the mean value
        Returns
        -------
        u : ndarray(d)
        """
        return self.eval_mem.mean(axis=0)
    
    def correlation(self,lags):
        """ Compute the corrleations at different lag times
        Parameters
        ----------
        lags : ndarray(k) or list of int
            lag times for computing the correlation. (The unit of lag times must be nonnegative.)

        Returns
        -------
        C_mem : ndarray(k,d,d)
            C_mem[i] is the correlation matrix at lag time lags[i]
        actual_lags : ndarray(k)
            the actual lag times for computing the correlations. Note it may be different with lags
            if lags contains negative values.
        """
        if np.isscalar(lags):
            tmp_lags=np.array([lags])
        else:
            tmp_lags=np.array(lags)
        actual_lags=np.maximum(np.rint(tmp_lags),0).astype(int)
        C_mem=_empirical_correlation_scan(actual_lags,self.eval_mem,self.traj_no_mem)
        if np.isscalar(lags):
            actual_lags=actual_lags[0]
            C_mem=C_mem[0]
        return C_mem,actual_lags

