import scipy.special as sps
import scipy.linalg as spl
import numpy as np
from scipy.interpolate import interp1d

def e2t(N,dE):
    return 2 * np.pi * np.fft.fftfreq(N,dE)

def gaussian_fun(fwhm):
    prefac = np.sqrt(2 * np.sqrt(np.log(2))/(np.sqrt(np.pi) * fwhm))
    alpha = 4.0 * np.log(2)/fwhm**2.0
    def g(x):
        return np.exp(- alpha * x**2.0)
    return g

def convert_fwhm(fwhm):
    return 8.0 * np.log(2)/fwhm

def thisfft(y):
    return np.fft.fft(y, n=len(y))

def ithisfft(y):
    return np.fft.ifft(y, n=len(y))


class Pulse:
    def __init__(self,
                 start, dbin, amps,
                 fwhm_tl, fwhm_len,
                 npts=2000,
                 ebounds=None, grid=None, w0=5.0):
        if grid is None:
            end = start + (len(amps)-1) * dbin
            self.ebins = np.array([start - dbin/2.0] +
                                  [start + i * dbin for i in range(len(amps))]
                                  + [end + dbin/2.0])
            self.amplitudes = np.array([amps[0]] + list(amps) + [amps[-1]])

        else:
            start =np.min(grid)
            end = np.max(grid)
            self.ebins = grid
            self.amplitudes = amps

        if ebounds is None:
            low = start - 5.0/fwhm_tl * 2 *np.pi
            high = end + 5.0/fwhm_tl * 2 *np.pi
        else:
            low = ebounds[0]
            high = ebounds[1]


        self.e = np.linspace(low, high, npts)
        self.amp = np.zeros(len(self.e),dtype=np.complex)
        indm = np.argmin(abs(self.e - self.ebins[0]))
        for i in range(1,len(self.ebins)):
            indp = np.argmin(abs(self.e - self.ebins[i]))
            self.amp[indm:indp] = self.amplitudes[i]
            indm = indp

        x = self.e - w0
        # The root 2 is for intensity as opposed to amplitude
        tl = gaussian_fun(convert_fwhm(fwhm_tl * np.sqrt(2)))
        total = gaussian_fun(convert_fwhm(fwhm_len * np.sqrt(2)))
        self.filt_tl = tl(x)
        self.filt_len = total(x)

        self.amp = np.convolve(self.amp * self.filt_tl,self.filt_len, 'same')
        self.dx = x[1] - x[0]
        self.w0 = w0

        self.t =  np.fft.fftshift(e2t(len(x), self.dx))
        self.amp = np.convolve(self.amp * self.filt_tl,self.filt_len, 'same')
        self.w0 = w0

    def get_grid(self, times):
        self.pt = np.fft.fftshift(thisfft(self.amp))
        grid = np.zeros((len(self.e),len(times)), dtype=np.complex)
        for i in range(len(times)):
            newp = self.pt.copy()
            newp[self.t>times[i]] = 0.0
            grid[:,i] = ithisfft(np.fft.ifftshift(newp))
        return grid

    def get_preps(self,times, e):
        preps = np.zeros((len(e),len(times)), dtype=np.complex)
        grid = self.get_grid(times)
        for i in range(len(times)):
            f = interp1d(self.e, grid[:,i], fill_value=0.0,bounds_error=False, kind='nearest')
            preps[:,i] = f(e)
        return preps

    def chirp(self, delay=0.0, GDD=0.0, TOD=0.0):
        # apply a chirp to the pulse
        planck = 4.13566751691e-15 # ev s
        hbarfs = planck * 1e15 / (2 * np.pi) #ev fs
        # in fs
        cent = self.w0
        phi_mat = (delay/hbarfs) * (self.e-cent) +\
                  (GDD/hbarfs**2.0) *  (self.e -cent)**2.0/2.0 +\
                  (TOD/hbarfs**3.0) * (self.e -cent)**3.0/6.0

        self.amp *= np.exp(1j*phi_mat)
