__author__ = 'jan-hendrikprinz'

import numpy as np

class ScalarProduct(object):
    """
    Represents the basis given the implementation of the used metric tensor.

    Notes
    -----

    """

    def dot(self, a, b):

        assert isinstance(a, Matrix), 'a needs to Matrix'
        assert isinstance(b, Matrix), 'b needs to Matrix'

        return Matrix()


class TimeScales(object):
    """
    A vector of physical Timescales
    """

    def __init__(self, l, u):
        self._lambda = l
        self._unit = u

    @classmethod
    def from_lambda(cls, l):
        obj = cls()
        obj._lambda = l

        return obj

    @classmethod
    def from_rate(cls, l):
        obj = cls()
        obj._lambda = 1.0 / l

    def __getitem__(self, item):
        return item

    def __call__(self, tau):

        obs = map(Process, range(len(self._lambda) ))

        return BilinearMatrix(
            obs_A = obs,
            matrix = np.diag(self._lambda ** tau / self._unit),
            obs_B = obs
        )


class Matrix(object):
    """
    Mixin for a tensor of 2nd order under a given basis
    """
    def __init__(self, obs_A, matrix, obs_B):
        self._matrix = matrix
#        self._basis = basis

        self._obs_A = obs_A
        self._obs_B = obs_B

    @property
    def obs_A(self):
        return self._obs_A

    @property
    def obs_B(self):
        return self._obs_B

    @property
    def m(self):
        """
        numpy representation in the current basis
        """
        return self._matrix

#    def b(self):
#        """
#        Return the basis
#        """
#        return self._basis

    def transform(self):
        """
        Convert to a different basis
        """
        return Matrix()

class BilinearMatrix(Matrix):
    """
    A matrix result from a Bilinear computation using specific Observables

    m_ij = BL(X_i, Y_j)
    """

    def __str__(self):
        if self.obs_A == self.obs_B:
            return 'Bilinear <%s,%s>' % (observable_list_to_str(self.obs_A), observable_list_to_str(self.obs_A))
        else:
            return 'Bilinear <%s,%s>' % (observable_list_to_str(self.obs_A), observable_list_to_str(self.obs_B))

class LinearMappingMatrix(Matrix):
    """
    A matrix result from a Linear transformation using specific Observables

    m_ij = < L * X_i , Y_i >

    Notes
    -----
    So, if A is a LinearMapping X_i -> Y_j and B is a bilinear B(X_i, X_j) we get

    (B | A)_ik = B(X_i, Y_k)

    """

    def __init__(self, obs_A, matrix, obs_B, forward=True):
        super(LinearMappingMatrix, self).__init__(obs_A, matrix, obs_B)
        self.forward = forward

    def __str__(self):
        if self.forward:
            return 'Linear %s > %s' % (observable_list_to_str(self.obs_A), observable_list_to_str(self.obs_B))
        else:
            return 'Linear %s < %s' % (observable_list_to_str(self.obs_B), observable_list_to_str(self.obs_A))

    @property
    def adj(self):
        return LinearMappingMatrix(
            obs_A = self._obs_A,
            obs_B = self._obs_B,
            matrix = self._matrix.T,
            forward=not self.forward
        )

    @property
    def left_obs(self):
        if self.forward:
            return self._obs_A
        else:
            return self._obs_B

    @property
    def right_obs(self):
        if self.forward:
            return self._obs_B
        else:
            return self._obs_A

    def __ror__(self, other):
        if isinstance(other, BilinearMatrix):
            if self._obs_A == other._obs_B:
                return BilinearMatrix(
                    obs_A = other._obs_A,
                    matrix = np.dot(other._matrix, self.matrix_forward),
                    obs_B = self._obs_B
                )
            else:
                raise TypeError('Incompatible observables')
        else:
            raise NotImplementedError()

    @property
    def matrix_forward(self):
        if self.forward:
            return self._matrix
        else:
            return self._matrix.T

    @property
    def matrix_backward(self):
        if self.forward:
            return self._matrix.T
        else:
            return self._matrix

    def __or__(self, other):
        if isinstance(other, LinearMappingMatrix):
            if self._obs_B == other._obs_A:
                return LinearMappingMatrix(
                    obs_A = self._obs_A,
                    matrix = np.dot(self.matrix_forward, other.matrix_forward),
                    obs_B = other._obs_B
                )
            else:
                raise NotImplementedError()
        elif isinstance(other, BilinearMatrix):
            if self._obs_A == other._obs_A:
                return BilinearMatrix(
                    obs_A = self._obs_B,
                    matrix = np.dot(self.matrix_backward, other._matrix),
                    obs_B= other._obs_B
                )
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()


class Model(object):
    def __init__(self):
        super(Model, self).__init__()
        self._correlation = None
        self._timescales = None

    def _get_correlation(self):
        raise NotImplementedError()

    def _get_timescales(self):
        raise NotImplementedError()

    @property
    def timescales(self):
        if self._timescales is None:
            # create new correlation object
            self._timescales = self._get_timescales()

        return self._timescales

    @property
    def correlation(self):
        if self._correlation is None:
            # create new correlation object
            self._correlation = self._get_correlation()

        return self._correlation


class StateModel(Model):
    def __init__(self):
        super(StateModel, self).__init__()
        self._propagator = None

    @property
    def propagator(self):
        if self._propagator is None:
            # create new correlation object
            self._propagator = self._get_propagator()

        return self._propagator

    @property
    def equilibrium(self):
        if self._eq is None:
            self._eq = self._get_eq()

        return self._eq

    def _get_eq(self):
        raise NotImplementedError()

    def _get_propagator(self):
        raise NotImplementedError()

    def sample(self, initial, length):
        return Realization()


class Correlation(Matrix):
    pass


class Propagator(Matrix):
    pass


class PropagatorModel(object):
    def __init__(self):
        pass

    def at(self, tau):
        """
        Return correlation matrix at concrete tau

        """
        return Propagator()

    def __call__(self, tau):
        return self.at(tau)


class CorrelationFunction(object):
    """
    General class for function expressible as Q L(tau) Q
    """

    def __init__(self, QL, timescale, QR):
        self._QR = QR
        self._QL = QL
        self._l = timescale

    def at(self, tau):
        """
        Return correlation matrix at concrete tau

        """
        return Correlation( self._QR | self._l(tau) | self._QL )

    def __call__(self, tau):
        return self.at(tau)


class MarkovStateModel(StateModel):
    def __init__(self, transition_matrix):
        super(MarkovStateModel, self).__init__()
        self._tm = transition_matrix
        self._eq = None
        self._correlation = None
        self._propagation = None

    def _get_eq(self):
        return

    def _get_timescales(self):
        return TimeScales()


class ProjectedMarkovStateModel(StateModel):
    pass


class ObservableOperatorModel(StateModel):
    pass

def observable_list_to_str(observables):
    s = ''
    typ = None
    n = None
    f = None
    finish = False
    seq = False
    for obs in observables:
        if type(obs) is not typ:
            if typ is not None:
                s += '),'

            s += obs.name + '('
            typ = type(obs)

            f = None

        if f is None:
            n = obs._n
            f = obs._n
            s += str(f)
        else:
            if type(obs._n) is int:
                if obs._n == n + 1:
                    n = n + 1
                    if s[-1] != '-':
                        s += '-'
                else:
                    s += str(n)
                    s += ','
                    f = obs._n
                    n = obs._n
                    s += str(f)
            else:
                s += ',' + obs._n

    if s[-1] == '-':
        s += str(n)

    s += ')'

    return s

class Observable(object):
    """
    A projection from space to an observable (R)
    """

    def __init__(self, name=None, n=None):
        self._name = name
        self._n = n

    def __str__(self):
        return '%s[%d]' % (self.name, self.n)

    @property
    def name(self):
        if self._name is not None:
            return self._name
        else:
            return 'Unknown'

    @property
    def n(self):
        return self._n

    def __eq__(self, other):
        if self.name == other.name:
            if self.n == other.n:
                return True

        return False

class Process(Observable):
    """
    Represents n - slowest process
    """
    def __init__(self, n):
        super(Process, self).__init__('Process', n)

class State(Observable):
    """
    Represents Observable State n
    """
    def __init__(self, n):
        super(State, self).__init__('State', n)


class StateSet(Observable):
    """
    Represents Observable Set of States [n_1, ..., n_k]
    """
    def __init__(self, list_n):
        super(StateSet, self).__init__('Set', list_n)


class FuzzyStateSet(Observable):
    def __init__(self, dist):
        super(FuzzyStateSet, self).__init__('Dist', n)


class Expectation(CorrelationFunction):
    pass

class Transformation(object):
    """
    A projection from one observable to another
    """

    def __init__(self):
        self.from_space = None
        self.to_space = None
        self._data = None

class Projection(object):
    """
    A projection from one space to another
    """

class Realization(object):
    """
    Attributes
    ----------
    data : the timeseries
    """
    def __init__(self):
        self._data = None

    @property
    def data(self):
        return self._data

    @classmethod
    def from_numpy(cls, data):
        obj = cls()
        obj._data = data