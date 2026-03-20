import os
import obspy
from seispy.io import Query
from obspy import UTCDateTime
import glob
import array
from obspy.taup import TauPyModel
from obspy.core.inventory import read_inventory
import numpy as np
import matplotlib.pyplot as plt
from obspy.core.trace import Trace
from joblib import parallel_config
from joblib import Parallel, delayed
from obspy.signal.util import next_pow_2
from scipy.fftpack import fft, ifft
from obspy.io.sac import SACTrace


#Functions taken from Seispy, edited to output the fit, so it can be used as a quality control parameter
def gaussFilter(dt, nft, f0):
    """
    Gaussian filter in frequency domain.

    :param dt: sample interval in second
    :type dt: float
    :param nft: number of samples
    :type nft: int
    :param f0: Gauss factor
    :type f0: float

    :return: Gaussian filter in frequency domain
    :rtype: np.ndarray
    """
    df = 1.0 / (nft * dt)
    nft21 = 0.5 * nft + 1
    f = df * np.arange(0, nft21)
    w = 2 * np.pi * f

    gauss = np.zeros([nft, 1])
    gauss1 = np.exp(-0.25 * (w / f0) ** 2) / dt
    gauss1.shape = (len(gauss1), 1)
    gauss[0:int(nft21)] = gauss1
    gauss[int(nft21):] = np.flipud(gauss[1:int(nft21) - 1])
    gauss = gauss[:, 0]

    return gauss

def gfilter(x, nfft, gauss, dt):
    """
    Apply Gaussian filter on time series.

    :param x: input trace
    :type x: np.ndarray
    :param nfft: number of samples
    :type nfft: int
    :param gauss: Gaussian filter in frequency domain
    :type gauss: np.ndarray
    :param dt: sample interval in second
    :type dt: float

    :return: Filtered data in time domain
    :rtype: np.ndarray
    """
    Xf = fft(x, nfft)
    Xf = Xf * gauss * dt
    xnew = ifft(Xf, nfft).real
    return xnew

def correl(R, W, nfft):
    """
    Correlation in frequency domain.

    :param R: numerator
    :type R: np.ndarray
    :param W: denominator
    :type W: np.ndarray

    :return: Correlation in frequency domain
    :rtype: np.ndarray
    """
    x = ifft(fft(R, nfft) * np.conj(fft(W, nfft)), nfft)
    x = x.real
    return x

def phaseshift(x, nfft, dt, tshift):
    """
    Phase shift in frequency domain.
    
    :param x: input trace
    :type x: np.ndarray
    :param nfft: number of samples
    :type nfft: int
    :param dt: sample interval in second
    :type dt: float
    :param tshift: Time shift before P arrival
    :type tshift: float

    :return: Phase shifted data in time domain
    :rtype: np.ndarray
    """
    Xf = fft(x, nfft)
    shift_i = int(tshift / dt)
    p = 2 * np.pi * np.arange(1, nfft + 1) * shift_i / nfft
    Xf = Xf * np.vectorize(complex)(np.cos(p), -np.sin(p))
    x = ifft(Xf, nfft) / np.cos(2 * np.pi * shift_i / nfft)
    x = x.real
    return x


def deconit(uin, win, dt, nt=None, tshift=10, f0=2.0, itmax=400, minderr=0.001, phase='P'):
    """
    Iterative deconvolution using Ligorria & Ammon method.
    @author: Mijian Xu @ NJU
    Created on Wed Sep 10 14:21:38 2014 

    :param uin: R or Q component for the response function
    :type uin: np.ndarray
    :param win: Z or L component for the source function
    :type win: np.ndarray
    :param dt: sample interval in second
    :type dt: float
    :param nt: number of samples, defaults to None
    :type nt: int, optional
    :param tshift: Time shift before P arrival, defaults to 10.
    :type tshift: float, optional
    :param f0: Gauss factor, defaults to 2.0
    :type f0: float, optional
    :param itmax: Max iterations, defaults to 400
    :type itmax: int, optional
    :param minderr: Min change in error required for stopping iterations, defaults to 0.001
    :type minderr: float, optional
    :param phase: Phase of the RF, defaults to 'P'
    :type phase: str, optional

    :return: (RFI, rms, it) RF, rms and number of iterations.
    :rtype: (np.ndarray, np.ndarray, int)
    """
    # print('Iterative Decon (Ligorria & Ammon):\n')
    if len(uin) != len(win):
        raise ValueError('The two input trace must be in same length')
    elif nt is None:
        nt = len(uin)
    else:
        pass

    rms = np.zeros(itmax)
    nfft = next_pow_2(nt)
    p0 = np.zeros(nfft)

    u0 = np.zeros(nfft)
    w0 = np.zeros(nfft)

    u0[0:nt] = uin
    w0[0:nt] = win

    gaussF = gaussFilter(dt, nfft, f0)
    # gaussF = _gauss_filter(dt, nfft, f0)

    u_flt = gfilter(u0, nfft, gaussF, dt)
    w_flt = gfilter(w0, nfft, gaussF, dt)

    wf = fft(w0, nfft)
    r_flt = u_flt

    powerU = np.sum(u_flt ** 2)

    it = 0
    sumsq_i = 1
    d_error = 100 * powerU + minderr
    maxlag = 0.5 * nfft
    # print('\tMax Spike Display is ' + str((maxlag) * dt))

    # while np.abs(d_error) > minderr and it < itmax:
    for it in range(itmax):
        rw = correl(r_flt, w_flt, nfft)
        rw = rw / np.sum(w_flt ** 2)

        if phase == 'P':
            i1 = np.argmax(np.abs(rw[0:int(maxlag) - 1]))
        else:
            i1 = np.argmax(np.abs(rw))
        amp = rw[i1] / dt

        p0[i1] = p0[i1] + amp
        p_flt = gfilter(p0, nfft, gaussF, dt)
        p_flt = gfilter(p_flt, nfft, wf, dt)

        r_flt = u_flt - p_flt
        sumsq = np.sum(r_flt ** 2) / powerU
        rms[it] = sumsq
        d_error = 100 * (sumsq_i - sumsq)
        sumsq_i = sumsq
        if np.abs(d_error) < minderr:
            break
        # it = it + 1

    p_flt = gfilter(p0, nfft, gaussF, dt)
    p_flt = phaseshift(p_flt, nfft, dt, tshift)
    RFI = p_flt[0:nt]
    rms = rms[0:it - 1]

    return RFI, rms, it, d_error

class RFTrace(obspy.Trace):
    """ 
    Class for receiver function trace.
    """
    def __init__(self, data=..., header=None):
        super().__init__(data=data, header=header)

    @classmethod
    def deconvolve(cls, utr, wtr, method='iter', tshift=10, f0=2.0, **kwargs):
            """
            Deconvolve to extract receiver function from waveforms.

            :param utr: R or Q component for the response function
            :type utr: ``obspy.Trace``
            :param wtr: Z or L component for the source function
            :type wtr: ``obspy.Trace``
            :param method: Method for deconvolution, defaults to 'iter'

            :type method: str, optional
            :param tshift: Time shift before P arrival, defaults to 10.
            :type tshift: float, optional
            :param f0: Gaussian factor, defaults to 2.0
            :type f0: float, optional
            :param kwargs: Parameters for deconvolution
            :type kwargs: dict

            :return: RFTrace object
            :rtype: RFTrace
            """
            header = utr.stats.__getstate__()
            for key, value in kwargs.items():
                header[key] = value
            if method.lower() == 'iter':
                rf, rms, it, d_error = deconit(utr.data, wtr.data, utr.stats.delta, tshift=tshift, f0=f0, **kwargs)
                header['rms'] = rms
                header['iter'] = it
                header['d_error']=d_error
            else:
                raise ValueError('method must be \'iter\' or \'water\'')
            header['tshift'] = tshift
            header['f0'] = f0
            
            rftr = cls(rf, header)
            return rftr


events=[]
model = TauPyModel(model='prem')

f0 = [1,2,3,4,5,6]
itmax = 400
minderr = 0.001
shift= 10
hello= True

fitmin = 0.60  # Minimum 60 percent of radial compoment should be fit (after reconvolving the RF with the vertical component

directory= ['/raid2/cg812/Good_2015_earthquakes/VIFE', '/raid2/cg812/Good_2015_earthquakes/LOGR', '/raid2/cg812/Good_2015_earthquakes/NAUG', '/raid2/cg812/Good_2015_earthquakes/DREK', '/raid2/cg812/Good_2015_earthquakes/HOTT', '/raid2/cg812/Good_2015_earthquakes/DYSA', '/raid2/cg812/Good_2015_earthquakes/STOR', '/raid2/cg812/Good_2015_earthquakes/VIKS']
def calculate_RF(station):
    rfstack=list()
    events= glob.glob(station + '/*[!.png]')
    for event in events:
        st= obspy.read(event)
        
        for i in f0:
            already= os.path.join('/raid2/cg812/Processed_RFs/Gauss_' + str(i) + '.0/', st[0].stats.station + '/' + st[0].stats.starttime.strftime("%Y%m%dT%H%M%S"))
            if not os.path.exists(already):    
                direc= '/raid2/cg812/Transverse_RFs/Gauss_' + str(i) + '.0/' + st[0].stats.station
                if not os.path.exists(direc):
                        os.makedirs(direc)
                #This can be modified to extract transverse receiver functions instead
                rf = RFTrace.deconvolve(st.select(channel='**R')[0], st.select(channel='**Z')[0], method='iter',
                                tshift=shift, f0 = i, itmax = 400, minderr = 0.001)
                new_trace = obspy.Trace(
                        data=rf.data,
                        header=rf.stats
                    )

                rf = obspy.Stream([new_trace])

                #shift it so all peaks align, not necessarily 10 as in the seispy code
                indm = np.argmax(np.abs(rf[0].data))
                rf.trim(rf[0].stats.starttime + indm/20 - 5, rf[0].stats.starttime + indm/20 + 60)

                fit= 1- rf[0].stats.d_error
                print(fit)
                indm = np.argmax(np.abs(rf[0].data)) 

        #Select reciever functions with good 
                
                
                if fit > fitmin:
                    savepath= os.path.join(direc, st[0].stats.starttime.strftime("%Y%m%dT%H%M%S"))
                    rf.write(savepath, format= 'PICKLE')
                    rf.plot(outfile=savepath)
                else:
                    print('Fit not large enough')


with parallel_config(backend= 'loky', n_jobs=1, verbose=5):
    Parallel()(delayed(calculate_RF)(station) for station in directory)


   
                        
