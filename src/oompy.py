'''
.. module:: oompy
   :synopsis: An observable operator module for python (17.09.2014)

.. moduleauthor:: Jan-Hendrik Prinz <jan.prinz@choderalab.org>
.. moduleauthor:: Hao Wu <hao.wu@fu-berlin.de>

'''

class OOMPy(object):
    '''A Python implementation for finite state observable operator models.
        
    Parameters
    ----------
    states : int
        number of internal states to be assumed. This will determine the dimensionality of the observable operators
    observables : int
        number of observable states. If None we assume an arbitrary number which is determined from the actual timeseries
                    
    Notes
    -----
    Might write an estimator for sklearn
    How to implement the binless version of OOMs.
    
    References
    ----------
    If you need more information read [Jaeger98]_ or refer to [Wu14]_ once it is published.

    .. [Jaeger98] Jaeger, H. Discrete Time, Discrete Valued Observable Operator Models: A Tutorial. (1998).
    .. [Wu14] Wu, H., Prinz, J.-H. & Noe, F. Observable operator models for metastable dynamics. In Preparation.    

    '''

    def __init__(self, states, observables = None):        
        '''
        Arguments
        ---------
        operators : numpy.ndarray, shape=(observable, state, states)
            the array actually storing the observable operators
        pre : numpy.ndarray, shape=(state, )
            the initial internal distribution, often referred to as `w`
        post : numpy.ndarray, shape=(state, )
            the final internal distribution. In classic OOM just the constant vector
        '''
        self.states = states
        self.observables = observables
        
    def learn(self, data):
        '''
        Estimate an OOM from a discrete time series
        
        Parameters
        ----------
        data : numpy.ndarray, shape=(length_of_timeseries)
            The actual timeseries in the observed space used as input data
        
        Notes
        -----
        For sklearn this would be called fit
        '''
        self.opreators = None
        self.pre = None
        self.post = None
        pass
    
    def path_probability(self, path_bundle):
        '''
        Returns the probabililty for the given path bundle under the previously estimated model
        
        Notes
        -----
        For sklearn we would call this predict and only allow for a single path.
        '''