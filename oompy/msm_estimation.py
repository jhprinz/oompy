
# coding: utf-8

# In[1]:

import numpy as np
import scipy.linalg as linalg
import numba

from numba import jit,int_,double

import pyemma.coordinates as coor
import pyemma.msm as msm

from pyemma import config
config['show_progress_bars']=False


# In[2]:

@jit(numba.typeof((np.array([[1.0]]),np.array([[[1.0]]])))(int_,int_[:],double[:,:]),nopython=True)
def _Markov_state_scan(state_num,dtraj_mem,eval_mem):
    T,eval_dim=eval_mem.shape
    N_list=np.zeros(state_num,dtype=np.float64)
    cen_list=np.zeros((state_num,eval_dim),dtype=np.float64)
    corr_list=np.zeros((state_num,eval_dim,eval_dim),dtype=np.float64)
    for t in range(T):
        k=dtraj_mem[t]
        N_list[k]+=1
        for i in range(eval_dim):
            cen_list[k,i]+=eval_mem[t,i]
            for j in range(eval_dim):
                corr_list[k,i,j]+=eval_mem[t,i]*eval_mem[t,j]
    for k in range(state_num):
        for i in range(eval_dim):
            cen_list[k,i]/=N_list[k]
            for j in range(eval_dim):
                corr_list[k,i,j]/=N_list[k]
    return cen_list,corr_list


# In[3]:

@jit(double[:,:](double[:],double[:,:,:]),nopython=True)
def _weighted_sum_matrices(w,C_mem):
    dim=C_mem.shape[1]
    N=w.shape[0]
    C=np.zeros((dim,dim),dtype=np.float64)
    for n in range(N):
        for i in range(dim):
            for j in range(dim):
                C[i,j]+=w[n]*C_mem[n,i,j]
    return C


# In[4]:

class MSMEstimator:

    def __init__(self,lag=1):
        """An MSM estimator for mean and correlations
        Parameters
        ----------
        lag : int
            lag time for dynamical modeling
        """
        self.lag=lag

    def estimate(self,trajs,eval_trajs,dmin):
        """ Estimate an MSM from data
        Parameters
        ----------
        trajs : ndarray(T,d0) or a list
            trajectories of corrdinates for constructing MSM
        eval_trajs : ndarray(T,d) or a list
            trajectories of quantities needed to be estimated
        dmin : double
            the threshold for regular space clustering
        """
        clr=coor.cluster_regspace(trajs,dmin)
        self.state_num=clr.clustercenters.shape[0]
        cen_list,corr_list=_Markov_state_scan(self.state_num,np.concatenate(clr.dtrajs),np.vstack(eval_trajs))
        self.model=msm.estimate_markov_model(clr.dtrajs,self.lag)
        active_set=self.model.active_set
        cen_list=cen_list[active_set]
        corr_list=corr_list[active_set]
        pii=self.model.stationary_distribution
        self.A_for_corr=cen_list.T*pii
        self.B_for_corr=cen_list.copy()
        self.C_for_corr=_weighted_sum_matrices(pii,corr_list)
    
    def expectation(self):
        """ Compute the mean value
        Returns
        -------
        u : ndarray(d)
        """
        return self.A_for_corr.sum(axis=1)
    
    def correlation(self,lags):
        """ Compute the corrleations at different lag times
        Parameters
        ----------
        lags : ndarray(k) or list of int
            lag times for computing the correlation. (The unit of lag times must be nonnegative
            and integral multiples of self.lag.)

        Returns
        -------
        C_mem : ndarray(k,d,d)
            C_mem[i] is the correlation matrix at lag time lags[i]
        actual_lags : ndarray(k)
            the actual lag times for computing the correlations. Note it may be different with lags
            if some elements of lags are negative or not integral multiples of self.lag.
        """
        P=self.model.P.copy()
        if np.isscalar(lags):
            actual_lag=max(int(np.rint((lags+0.0)/self.lag)),0)
            if actual_lag==0:
                return self.C_for_corr,actual_lag*self.lag
            C=self.A_for_corr.dot(np.linalg.matrix_power(P,actual_lag)).dot(self.B_for_corr)
            return 0.5*(C+C.T),actual_lag*self.lag
        else:
            lags=np.array(lags)
            actual_lags=np.maximum(np.rint((lags+0.0)/self.lag),0).astype(int)
            C_mem=np.empty(actual_lags.shape+self.C_for_corr.shape)
            ad=0
            AD=self.A_for_corr.copy()
            d=0
            D=np.identity(P.shape[0])
            for n in range(actual_lags.shape[0]):
                i=actual_lags[n]
                if i-ad==d:
                    AD=AD.dot(D)
                    ad=i
                    C=AD.dot(self.B_for_corr)
                elif i-ad>=0:
                    d=i-ad
                    D=np.linalg.matrix_power(P,d)
                    AD=AD.dot(D)
                    ad=i
                    C=AD.dot(self.B_for_corr)
                else:
                    ad=i
                    AD=self.A_for_corr.dot(np.linalg.matrix_power(P,i))
                    d=0
                    D=np.identity(P.shape[0])
                    C=AD.dot(self.B_for_corr)
                if i==0:
                    C=self.C_for_corr.copy()
                C_mem[n]=0.5*(C+C.T)
            
        return C_mem,actual_lags*self.lag

