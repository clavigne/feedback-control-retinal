import numpy as np
from PulseGenerator import Pulse

# physical constants
planck = 4.13566751691e-15 # ev s
hbarfs = planck * 1e15 / (2 * np.pi) #ev fs
ev_nm = 1239.842
opt_t = np.linspace(900,1100,10)/hbarfs

def build_fitness_function(
        nbins=30,
        tl_duration=19.0,
        e_carrier=2.22,
        e_shaper=0.3,
        total_duration=1000.0,
        avgtimes=[1200.0],):

    # Load necessary quantites
    es = np.load("operators/es.npy")
    fcf_e = np.load("operators/fcf_e.npy")
    Pce = np.load("operators/Pce.npy")
    Pte = np.load("operators/Pte.npy")

    # Ultrafast laser parameters
    width_tl = tl_duration/hbarfs
    width_total = total_duration/hbarfs
    w0 = e_carrier
    elow = e_carrier - e_shaper/2.0
    ehigh = e_carrier + e_shaper/2.0

    de = (ehigh - elow)/nbins
    opt_times = np.array(avgtimes)/hbarfs

    def make_shaped_pulse(veci, veca):
        intensities = veci
        angles = veca

        intensities[intensities<0] = 0.0
        intensities[intensities>1] = 1.0

        intensities[angles<0] = 0.0
        intensities[angles>2*np.pi] = 2 * np.pi

        amplitudes = np.sqrt(intensities) * np.exp(1j * angles)
        gp = Pulse(elow, de, amplitudes, width_tl, width_total, w0=w0)
        return gp

    def prop_with_pulse(gp, times):
        preps = gp.get_preps(times,es)
        for i in range(len(es)):
            preps[i,:] *= np.exp(-1j * es[i] * times) * fcf_e[i] * (-1.0/1j)

        pt = np.array([psi.T.conj().dot(Pte).dot(psi) for psi in preps.T]).real
        pc = np.array([psi.T.conj().dot(Pce).dot(psi) for psi in preps.T]).real
        return pt, pc

    def evaluate(veci,veca):
        g = make_shaped_pulse(veci,veca)
        pt, pc = prop_with_pulse(g, opt_times)
        return np.sum(pt)/np.sum(pc + pt)

    return evaluate

if __name__ == "__main__":
    # Build a fitness function that represents shaped pulse control of
    # cis-trans isomerization of retinal in bacteriorhodopsin. The paremeters
    # below are those used in the paper:
    #         https://aip.scitation.org/doi/abs/10.1063/1.5003389

    # nbins is the number of pixels on the pulse shaper. The final fitness
    # function has nbins x 2 dimensions:

    # - nbins values between 0 and 1 that give the transparency of the shaper
    # at that pixel.

    #- nbins angles (periodic, 0 to 2 pi radians) that give the delay applied
    # by the shaper at that pixel.

    # Values outside those bounds are truncated.


    # Parameters
    # -------------------------------------------------------------------------
    # nbins: number of controllable elements on the pulse shaper

    # tl_duration: duration of the seed pulse to the shaper, in fs.

    # e_carrier: carrier energy of the seed pulse, in eV.

    # e_shaper: bandwidth of the shaper, in eV.

    # total_duration: soft limit on the total pulse length, in fs.

    # avgtimes: time points at which the isomer ratio is computed and averaged
    # to obtain the final fitness ("control interval").

    fitness = build_fitness_function(
        nbins=30,
        tl_duration=19.0,
        e_carrier=2.22,
        e_shaper=0.3,
        total_duration=1000.0,
        avgtimes=np.linspace(900,1100,10))


    # evaluate some pulses
    values = [fitness(np.random.rand(30), 2 * np.pi * np.random.rand(30))
              for k in range(6)]


