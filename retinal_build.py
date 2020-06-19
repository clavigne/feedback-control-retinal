import numpy as np
import scipy.linalg as npl
import Operators as ops

# Constants
planck = 4.13566751691e-15 # ev s
hbarfs = planck * 1e15 / (2 * np.pi) #ev fs
ev_nm = 1239.842

# Operation flags
compute_hamiltonian = True
compute_operators = True


# -------------- Retinal 2-state 2-mode Hamiltonian ---------

# Define system and parameters
# ----------------------------

space = ops.System()
space.define_mode('q',  20)
space.define_mode('phi', 300, 'grid')
space.define_mode('el',  2, 'electronic')
space.consts = dict(wq  =  0.19,
                    minv =  4.84e-4,
                    E1   = 0.00,
                    E2   = 2.48,
                    k1= 0.00,
                    k2= 0.10,
                    W0 =3.6,
                    W1 = 1.09,
                    lbda =  0.19)

# Build Hamiltonian terms
# -----------------------
# Harmonic zeroth order
H0= [space.build_operator({ 'q' :'wq * q**2/2 + wq * p**2/2'}),
    space.build_operator({ 'phi':'minv * dx2/2.0'})]
# Linear shifts
H1 = [space.build_operator({'q':'k1 * q', 'el':'S0S0'}),
      space.build_operator({'q':'k2 * q', 'el':'S1S1'})]
# Torsion shifts
space.modes['phi'].define_term('cos', np.cos)
H1 += [
    space.build_operator({'phi':'V(0.5 *  W0 * (1.0 - cos(x)))',
                          'el':'S0S0'}),
    space.build_operator({'phi':'V(E2 -  0.5 * W1 * (1.0 - cos(x)))',
                          'el':'S1S1'})]
# Coupling
V = [space.build_operator({'q':'lbda * q', 'el':'S1S0'}),
     space.build_operator({'q':'lbda * q', 'el':'S0S1'})]
H = H0 + H1 + V
# ------------------------------------------------------------

if compute_hamiltonian:
    # Compute full, dense Hamiltonian matrix
    Hfull = H[0].to_fullmat()
    for i in range(1,len(H)):
        Hfull = Hfull + H[i].to_fullmat()
    Hfull = np.real(Hfull.todense())
    e, P = npl.eigh(Hfull)

    # Shift Hamiltonian energies (excitation is from 0.0 eV)
    absmax = 2.56                     # simulation absorption maximum
    eabsmax = ev_nm/565.0             # experimental absorption maximum
    shift = absmax - eabsmax
    e = e - shift

    np.save("matrices/diag_e.npy", e)
    np.save("matrices/diag_p.npy", P)


e = np.load("matrices/diag_e.npy")
P = np.load("matrices/diag_p.npy")

if compute_operators:
    x = space.modes['phi'].grid
    # Define cis-trans
    v2 = np.diag(H1[3].terms['phi'])
    mid = 0.5*(np.max(v2) + np.min(v2))
    lind = np.argmin(abs(mid - v2))
    rind = np.argmin(abs(mid - v2)[lind+1:]) + lind+1
    cis_l = x[lind]
    cis_r = x[rind]
    def cis(x):
        out = np.zeros(len(x))
        out[np.logical_not(np.logical_and(x>cis_l, x <= cis_r))] = 1.0
        return out
    space.modes['phi'].define_term('cis', cis)

    Pc = np.array(space.build_operator({'phi':'V(cis(x))'}
                                       ).to_fullmat().todense())
    Pt = np.array(space.build_operator({'phi':'V(1.0-cis(x))'}
                                       ).to_fullmat().todense())
    Pte = P.T.dot(Pt).dot(P)
    Pce = P.T.dot(Pc).dot(P)

    # FC state
    switch = np.array(space.build_operator({'el':'S1S0'}).to_fullmat().todense())
    gs = P[:,0]
    fcf = switch.dot(gs)
    fcf_e = P.T.dot(fcf)

    # Remove states that do not absorb
    sel = abs(fcf_e)>1e-7
    Pce = Pce[sel,:][:,sel]
    Pte = Pte[sel,:][:,sel]
    es = e[sel]
    fcf_e = fcf_e[sel]

    np.save('operators/fcf_e.npy', fcf_e)
    np.save('operators/Pte.npy',Pte)
    np.save('operators/Pce.npy',Pce)
    np.save('operators/es.npy', es)

