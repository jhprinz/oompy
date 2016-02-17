
# coding: utf-8

# In[1]:

import numpy as np
import scipy.linalg as linalg
import numba

from numba import jit,int_,double
from math import exp


# In[2]:

# return U,S,V   A=U*S*V'
def truncated_svd(A,m=np.inf):
    m=min(m,A.shape[0])
    U,s,Vh=linalg.svd(A)
    tol=A.shape[0]*np.spacing(s[0])
    m=min(m,np.count_nonzero(s>tol))
    return U[:,:m],np.diag(s[:m]),Vh[:m].T

# return U,S   A=U*S*V'
def truncated_svd_psd(A,m=np.inf):
    m=min(m,A.shape[0])
    S,U=linalg.schur(A)
    s=np.diag(S)
    tol=A.shape[0]*np.spacing(s.max())
    m=min(m,np.count_nonzero(s>tol))
    idx=(-s).argsort()[:m]
    return U[:,idx],np.diag(s[idx])

# return pinv(A)
def pinv_psd(A,m=np.inf):
    U,S=truncated_svd_psd(A)
    return U.dot(np.diag(1.0/np.diag(S))).dot(U.T)

# return R   A=R*R'
def cholcov(A,m=np.inf):
    U,S=truncated_svd_psd(A,m)
    return U.dot(np.sqrt(S))

# return pinv(R)   A=R*R'
def pinv_cholcov(A,m=np.inf):
    U,S=truncated_svd_psd(A,m)
    return np.diag(1.0/np.sqrt(np.diag(S))).dot(U.T)


# In[3]:

# return the empirical sum of f,C0,C1,C2, and normalize C0
@jit(numba.typeof((np.array([1.0]),np.array([[1.0]]),np.array([[1.0]]),np.array([[1.0]])))(double[:,:],double[:],int_,int_,double[:,:],numba.typeof([1])),nopython=True)
def _oom_model_scan(W,b,L,lag,traj_mem,traj_no_mem):
    T,dim=traj_mem.shape
    basis_num=W.shape[0]
    f=np.zeros(basis_num,dtype=np.float64)
    C0=np.zeros((basis_num,basis_num),dtype=np.float64)
    C1=np.zeros((basis_num,basis_num),dtype=np.float64)
    C2=np.zeros((basis_num,basis_num),dtype=np.float64)
    N=0
    
    f_n=np.empty(basis_num, dtype=np.float64)
    f_0=np.empty(basis_num, dtype=np.float64)
    f_p=np.empty(basis_num, dtype=np.float64)
    for t in range(L*lag,T-L*lag):
        if traj_no_mem[t-L*lag]==traj_no_mem[t+L*lag]:
            N+=1
            for n in range(basis_num):
                f_n[n]=b[n]
                f_0[n]=b[n]
                f_p[n]=b[n]
                k=0
                for i in range(t-L*lag,t,lag):
                    for j in range(dim):
                        f_n[n]+=W[n,k]*traj_mem[i,j]
                        f_0[n]+=W[n,k]*traj_mem[2*t-lag-i,j]
                        f_p[n]+=W[n,k]*traj_mem[2*t-i,j]
                        k+=1
                f_n[n]=exp(-f_n[n]*f_n[n]/2.0)
                f_0[n]=exp(-f_0[n]*f_0[n]/2.0)
                f_p[n]=exp(-f_p[n]*f_p[n]/2.0)
            for i in range(basis_num):
                f[i]+=f_n[i]
                for j in range(basis_num):
                    C0[i,j]+=f_n[i]*f_n[j]+f_0[i]*f_0[j]
                    C1[i,j]+=f_n[i]*f_0[j]
                    C2[i,j]+=f_n[i]*f_p[j]
                    
    for i in range(basis_num):
        for j in range(basis_num):
            C0[i,j]/=2.0*N
    
    return f,C0,C1,C2


# In[4]:

#The same with _oom_model_scan, only for test
def _oom_model_scan_numpy(W,b,L,lag,traj_mem,traj_no_mem):
    T,dim=traj_mem.shape
    basis_num=W.shape[0]
    f=np.zeros(basis_num,dtype=np.float64)
    C0=np.zeros((basis_num,basis_num),dtype=np.float64)
    C1=np.zeros((basis_num,basis_num),dtype=np.float64)
    C2=np.zeros((basis_num,basis_num),dtype=np.float64)
    N=0
    
    for t in range(L*lag,T-L*lag):
        if traj_no_mem[t-L*lag]==traj_no_mem[t+L*lag]:
            N+=1
            f_n=W.dot(traj_mem[range(t-L*lag,t,lag)].reshape(-1))+b
            f_0=W.dot(traj_mem[range(t+(L-1)*lag,t-lag,-lag)].reshape(-1))+b
            f_p=W.dot(traj_mem[range(t+L*lag,t,-lag)].reshape(-1))+b
            f_n=np.exp(-f_n*f_n/2)
            f_0=np.exp(-f_0*f_0/2)
            f_p=np.exp(-f_p*f_p/2)
            f+=f_n
            C0+=np.outer(f_n,f_n)+np.outer(f_0,f_0)
            C1+=np.outer(f_n,f_0)
            C2+=np.outer(f_n,f_p)

    C0/=2*N
    
    return f,C0,C1,C2


# In[5]:

@jit(numba.typeof((np.array([[1.0]]),)*3)(double[:,:],double[:],int_,int_,double[:,:],double[:,:],double[:],double[:],double[:,:],numba.typeof([1]),double[:,:]),nopython=True)
# Note: F1'*f_n*f_p*F2 is the observable operator, and F1'*f is the sigma
# The mean value is A*sigma, the correlation is C, and the correlation with lag k is A*Xi_O^(k-1)*B
def _oom_quantity_scan(W,b,L,lag,F1,F2,omega,sigma,traj_mem,traj_no_mem,eval_mem):
    T,dim=traj_mem.shape
    eval_dim=eval_mem.shape[1]
    basis_num=W.shape[0]
    order_num=omega.shape[0]
    
    A=np.zeros((eval_dim,order_num),dtype=np.float64)
    B=np.zeros((order_num,eval_dim),dtype=np.float64)
    C=np.zeros((eval_dim,eval_dim),dtype=np.float64)

    f_n=np.empty(basis_num,dtype=np.float64)
    f_p=np.empty(basis_num,dtype=np.float64)
    F1_fn=np.empty(order_num,dtype=np.float64) #F1'*f_n
    F2_fp=np.empty(order_num,dtype=np.float64) #F2'*f_p
    for t in range(L*lag,T-L*lag):
        if traj_no_mem[t-L*lag]==traj_no_mem[t+L*lag]:
            for n in range(basis_num):
                f_n[n]=b[n]
                f_p[n]=b[n]
                k=0
                for i in range(t-L*lag,t,lag):
                    for j in range(dim):
                        f_n[n]+=W[n,k]*traj_mem[i,j]
                        f_p[n]+=W[n,k]*traj_mem[2*t-i,j]
                        k+=1
                f_n[n]=exp(-f_n[n]*f_n[n]/2.0)
                f_p[n]=exp(-f_p[n]*f_p[n]/2.0)

            for i in range(order_num):
                F1_fn[i]=0
                F2_fp[i]=0
                for j in range(basis_num):
                    F1_fn[i]+=F1[j,i]*f_n[j]
                    F2_fp[i]+=F2[j,i]*f_p[j]
            
            omega_F1_fn=0
            for i in range(order_num):
                omega_F1_fn+=omega[i]*F1_fn[i]
            for i in range(eval_dim):
                for j in range(order_num):
                    A[i,j]+=eval_mem[t,i]*F2_fp[j]*omega_F1_fn
            
            fp_F2_sigma=0
            for i in range(order_num):
                fp_F2_sigma+=sigma[i]*F2_fp[i]
            for i in range(order_num):
                for j in range(eval_dim):
                    B[i,j]+=eval_mem[t,j]*F1_fn[i]*fp_F2_sigma
                    
            s=omega_F1_fn*fp_F2_sigma
            for i in range(eval_dim):
                for j in range(eval_dim):
                    C[i,j]+=s*eval_mem[t,i]*eval_mem[t,j]

    return A,B,C


# In[6]:

#The same with _oom_quantity_scan, only for test
def _oom_quantity_scan_numpy(W,b,L,lag,F1,F2,omega,sigma,traj_mem,traj_no_mem,eval_mem):
    T,dim=traj_mem.shape
    eval_dim=eval_mem.shape[1]
    basis_num=W.shape[0]
    order_num=omega.shape[0]
    
    A=np.zeros((eval_dim,order_num))
    B=np.zeros((order_num,eval_dim))
    C=np.zeros((eval_dim,eval_dim))

    for t in range(L*lag,T-L*lag):
        if traj_no_mem[t-L*lag]==traj_no_mem[t+L*lag]:
            f_n=(W.dot(traj_mem[range(t-L*lag,t,lag)].reshape(-1))+b).reshape(-1,1)
            f_p=(W.dot(traj_mem[range(t+L*lag,t,-lag)].reshape(-1))+b).reshape(-1,1)
            f_n=np.exp(-f_n*f_n/2)
            f_p=np.exp(-f_p*f_p/2)

            a=eval_mem[t].reshape(-1,1)
            
            Xi=F1.T.dot(f_n).dot(f_p.T).dot(F2)
            
            A+=a.dot(omega.reshape(1,-1)).dot(Xi)
            B+=Xi.dot(sigma.reshape(-1,1)).dot(a.T)
            C+=a.dot(omega.reshape(1,-1)).dot(Xi).dot(sigma.reshape(-1,1)).dot(a.T)

    return A,B,C


# In[7]:

class OOMEstimator:
    
    def __init__(self,lag=1,order_num=10,L=3,basis_num=100,normalization=True):
        """An OOM estimator for mean and correlations
        Parameters
        ----------
        lag : int
            lag time for dynamical modeling
        order_num : int
            the model order (dimension of observable operator matrices)
        L : int
            length of subsequences for estimation
        basis_num : int
            number of feature functions
        normalization : bool
            if normalization is True, 1 is an eigenvalue of Xi_O
        """
        self.lag=lag
        self.order_num=order_num
        self.L=L
        self.basis_num=basis_num
        self.normalization=normalization
    
    def _parameter_calculate(self,pii,C0,C1,C2):
        A=pinv_cholcov(C0)
        [U_m,Sigma_m,V_m]=truncated_svd(A.dot(C1).dot(A.T),self.order_num)
        self.order_num=min(self.order_num,Sigma_m.shape[0])
        self.F1=linalg.solve(Sigma_m,U_m.T.dot(A)).T
        self.F2=A.T.dot(V_m)
        sigma=self.F1.T.dot(pii.reshape(-1,1))
        self.Xi_O=self.F1.T.dot(C2).dot(self.F2)
        I_m=np.identity(self.order_num)
        if self.normalization is True:
            tmp_sigma=self.Xi_O.dot(sigma)
            D=(sigma-tmp_sigma).dot(linalg.pinv2(tmp_sigma))
            self.F1+=self.F1.dot(D.T)
            self.Xi_O+=D.dot(self.Xi_O)
        pinv_sigma=linalg.pinv2(sigma)
        Im_sigma=I_m-sigma.dot(pinv_sigma)
        omega=pinv_sigma-pinv_sigma.dot(self.Xi_O-I_m).dot(linalg.pinv2((
                    Im_sigma).dot(self.Xi_O-I_m))).dot(
                    Im_sigma)
        self.sigma=sigma.reshape(-1)
        self.omega=omega.reshape(-1)
    
    def estimate(self,trajs,eval_trajs):
        """ Estimate an OOM from data
        Parameters
        ----------
        trajs : ndarray(T,d0) or a list
            trajectories of corrdinates for constructing MSM
        eval_trajs : ndarray(T,d) or a list
            trajectories of quantities needed to be estimated
        """
        if type(trajs) is list:
            traj_no_mem=[]
            for i in range(len(trajs)):
                traj_no_mem+=[i]*trajs[i].shape[0]
        else:
            traj_no_mem=[0]*trajs.shape[0]
        traj_mem=np.vstack(trajs)
        eval_mem=np.vstack(eval_trajs)
        
        self.dim=traj_mem.shape[1]
        tmp_W=np.random.random((self.basis_num,self.L*self.dim))*2.0-1.0
        tmp_b=np.random.random(self.basis_num)
        tmp_mean=np.tile(np.mean(traj_mem,axis=0),self.L)
        tmp_std=np.tile(np.maximum(np.std(traj_mem,axis=0,ddof=1),1e-10),self.L)
        tmp_W=tmp_W/tmp_std
        tmp_b-=tmp_W.dot(tmp_mean)
        self.basis_W=tmp_W.copy()
        self.basis_b=tmp_b.copy()
        
        f,C0,C1,C2=_oom_model_scan(self.basis_W,self.basis_b,self.L,self.lag,traj_mem,traj_no_mem)
        self._parameter_calculate(f,C0,C1,C2)
        self.A_for_corr,self.B_for_corr,self.C_for_corr=_oom_quantity_scan(self.basis_W,self.basis_b,self.L,self.lag,
                                                                          self.F1,self.F2,self.omega,self.sigma,
                                                                          traj_mem,traj_no_mem,eval_mem)

    def expectation(self):
        """ Compute the mean value
        Returns
        -------
        u : ndarray(d)
        """
        return self.A_for_corr.dot(self.sigma)
    
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
        if np.isscalar(lags):
            actual_lag=max(int(np.rint((lags+0.0)/self.lag)),0)
            if actual_lag==0:
                return self.C_for_corr,actual_lag*self.lag
            C=self.A_for_corr.dot(np.linalg.matrix_power(self.Xi_O,actual_lag-1)).dot(self.B_for_corr)
            return 0.5*(C+C.T),actual_lag*self.lag
        else:
            lags=np.array(lags)
            actual_lags=np.maximum(np.rint((lags+0.0)/self.lag),0).astype(int)
            C_mem=np.empty(actual_lags.shape+self.C_for_corr.shape)
            ad=0
            AD=self.A_for_corr.copy()
            d=0
            D=np.identity(self.order_num)
            for n in range(actual_lags.shape[0]):
                i=actual_lags[n]
                if i-1-ad==d:
                    AD=AD.dot(D)
                    ad=i-1
                    C=AD.dot(self.B_for_corr)
                elif i-1-ad>=0:
                    d=i-1-ad
                    D=np.linalg.matrix_power(self.Xi_O,d)
                    AD=AD.dot(D)
                    ad=i-1
                    C=AD.dot(self.B_for_corr)
                elif i==0:
                    ad=0
                    AD=self.A_for_corr.copy()
                    d=0
                    D=np.identity(self.order_num)
                    C=self.C_for_corr.copy()
                else:
                    ad=i-1
                    AD=self.A_for_corr.dot(np.linalg.matrix_power(self.Xi_O,i-1))
                    d=0
                    D=np.identity(self.order_num)
                    C=AD.dot(self.B_for_corr)
                C_mem[n]=0.5*(C+C.T)
            return C_mem,actual_lags*self.lag

