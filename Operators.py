import numpy as np
import numpy.linalg as npl
import scipy.sparse as sps
from functools import reduce

# This is the max order of a polynomial potential we will be
# declaring. Basically the x and p matrices internally will be nbasis
# + MAX_ORDER so that q^MAX_ORDER is correct up to nbas.
MAX_ORDER = 4

class System:
    def __init__(self):
        self.modes = {}
        self.nmodes = 0
        self.constants = {}
        self.mode_list = []

        self.shape = []

    def define_mode(self, name, nbas, typ='harmonic', **params):
        if typ=='harmonic':
            self.modes[name] = HarmMode(name, nbas)
            self.mode_list += [name]
            self.shape += [nbas]
            self.nmodes+=1
        elif typ=='electronic':
            self.modes[name] = ElecMode(name, nbas)
            self.mode_list += [name]
            self.shape += [nbas]
            self.nmodes+=1
        elif typ=='grid':
            self.modes[name] = GridMode(name, nbas,**params)
            self.mode_list += [name]
            self.shape += [nbas]
            self.nmodes+=1
        else:
            raise NotImplemented("Type " + typ + " not implemented")

    def build_operator(self, dictionary):
        op = Operator(self)
        for key, val in dictionary.items():
            m = self.modes[key].interpret(val, self.consts)
            op.add_term(key, m)
        return op

    def get_index(self,index_set):
        ind = 0
        siz = [1]
        for key in self.mode_list[::-1]:
            ind = ind + index_set[key] * np.product(siz)
            siz += [self.modes[key].nbas]
        return ind

    def trace(self, matrix, modes):
        tensor = matrix.reshape(self.shape+self.shape)
        nmode_list = self.mode_list[:]
        for mode in modes:
            il = nmode_list.index(mode)
            ir = il + len(nmode_list)
            tensor = np.trace(tensor, axis1=il, axis2=ir)
            nmode_list.pop(il)
        return tensor


class Operator:
    def __init__(self, system):
        self.parent = system
        self.terms = {}
        for key in self.parent.modes.keys():
            self.terms[key] = None

    def add_term(self,mode, matrix):
        if type(matrix) is np.ndarray:
            self.terms[mode] = matrix
        else:
            self.terms[mode] = np.array(matrix.todense())

    def get_term(self, key):
        return self.terms.get(key, None)

    def to_fullmat(self):
        # For testing, returns the full matrix representation of the operator
        mats = []
        for name in self.parent.mode_list:
            val = self.terms[name]
            if val is None:
                mats += [self.parent.modes[name].get_I()]
            else:
                mats += [val]
        return reduce(lambda x,y:sps.kron(x,y,format='csr'), mats)

    def to_fullterms(self):
        # For testing, returns the full matrix representation of the operator
        mats = []
        for name in self.parent.mode_list:
            val = self.terms[name]
            if val is None:
                pass
            else:
                mats += [val]
        return mats

    def term_inv(self):
        new = Operator(self.parent)
        for k,v in self.terms.items():
            if v is not None:
                new.terms[k] = npl.inv(v)
        return new

class Mode:
    def interpret(self, string, consts):
        m = eval(string, consts, self.mode_consts)
        return m[:self.nbas,:self.nbas]

    def get_I(self):
        return sps.eye(self.nbas,self.nbas, format='csr')


class HarmMode(Mode):
    def __init__(self, name, nbas):
        self.name = name
        self.nbas = nbas
        self.mode_consts= dict(q=self.get_q(),
                               p=self.get_p(),
                               I=self.get_harm_I())

    def get_q(self, off=MAX_ORDER):
        """Return q operator."""
        i = np.arange(0, self.nbas+off-1)
        j = np.arange(1, self.nbas+off)
        dat = np.sqrt(0.5 * (i +1))
        mat = sps.csr_matrix( (dat,[i,j]), shape=(self.nbas+off, self.nbas + off))
        return mat + mat.T

    def get_p(self, off=MAX_ORDER):
        """Return laplacian operator."""
        i = np.arange(0, self.nbas+off-1)
        j = np.arange(1, self.nbas+off)
        dat = 1j * np.sqrt(0.5 * (i +1))
        mat = sps.csr_matrix( (dat,[i,j]), shape=(self.nbas+off, self.nbas + off))
        return mat - mat.T

    def get_harm_I(self, off=MAX_ORDER):
        return sps.eye(self.nbas+off,self.nbas+off, format='csr')

    def get_grid_points(self):
        xi, Pi = npl.eigh(self.get_q(off=0).todense())
        return xi, Pi

class ElecMode(Mode):
    def __init__(self, name, nbas):
        self.name = name
        self.nbas = nbas
        self.mode_consts = {}
        for i in range(nbas):
            for j in range(nbas):
                key = "S%iS%i"%(i,j)
                val = self.get_el(i,j)
                self.mode_consts[key] = val

    def get_el(self, iel, jel):
        ij = [[iel], [jel]]
        dat = [1.0]
        return sps.csr_matrix( (dat,ij), shape=(self.nbas, self.nbas))

class GridMode(Mode):
    def __init__(self, name, nbas, low=0.0, high=2*np.pi):
        self.name = name
        self.nbas = nbas
        self.bound = [low, high]
        self.grid = np.linspace(low, high, nbas)
        self.delta = self.grid[1] - self.grid[0]
        self.mode_consts = dict(dx2=self.get_lapl(),
                                I=self.get_I(),
                                V=np.diag,
                                x=self.grid)

    def get_lapl(self):
        T = np.zeros((self.nbas,self.nbas), dtype=np.complex)

        # Calculate prefactors and stuff
        k_squared = (np.pi/self.delta)**2.0
        n_squared = self.nbas**2.0
        diagonalElements = (k_squared/3.0)*(1.0+(2.0/n_squared))
        notDiagonal = (2*k_squared / n_squared)

        fun = (lambda i,j: # DO NOT CHANGE THIS (Formula from Tannor)
               diagonalElements if i==j
               else notDiagonal * (
                       np.power(-1.0,j-i) * np.power(np.sin(np.pi*(j-i)/self.nbas),-2)))

        for i in range(self.nbas):
            for j in range(self.nbas):
                T[i,j] = fun(i,j)
        return T

    def define_term(self,name, fun):
        self.mode_consts[name] = fun
