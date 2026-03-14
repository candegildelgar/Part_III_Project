from scipy.fft import fft, fftfreq
import numpy as np
import obspy
from obspy.taup import TauPyModel
from seispy.decon import RFTrace
from obspy.core.inventory import read_inventory

model = TauPyModel(model='iasp91')
xml_file= read_inventory('/raid4/tpo21/scripts/tom_mega_comb_april_24.xml')
f0=[1,2,3,4,5,6]

tr= obspy.read('/raid2/cg812/Raw_data/APAL/20240808T074255.CHE.PICKLE')
tr+= obspy.read('/raid2/cg812/Raw_data/APAL/20240808T074255.CHZ.PICKLE')
tr+= obspy.read('/raid2/cg812/Raw_data/APAL/20240808T074255.CHN.PICKLE')
tr.resample(20)
tr.remove_response(xml_file)

tr.detrend('linear') #detrend
        

        #For applying a cosine taper could use obspy.simulate_seismometer? Would check with Tom
tr.taper(max_percentage= 0.05, type='cosine')
tr.filter('highpass', freq=0.05)
st_TRZ = tr.copy().rotate('NE->RT', back_azimuth= tr[0].stats['baz'])
        
            # Check rotation worked

arrivals = model.get_travel_times(st_TRZ[0].stats['evdp'], st_TRZ[0].stats['dist'], phase_list=['P'])
P_arr = arrivals[0]

st_TRZ.trim(st_TRZ[0].stats.starttime+P_arr.time-10,
st_TRZ[0].stats.starttime+P_arr.time+ 120)

for i in f0:
        tr = RFTrace.deconvolute(st_TRZ.select(channel='**R')[0], st_TRZ.select(channel='**Z')[0], method='iter',
                                tshift=10, f0 =i, itmax = 400, minderr = 0.001)


        t= np.linspace(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.npts, endpoint='False')
        y= tr.data
        yf = fft(y)
        xf = fftfreq(tr.stats.npts, 1/tr.stats.sampling_rate)

        pos_mask = xf >= 0
        xf = xf[pos_mask]
        yf = np.abs(yf[pos_mask])

        threshold = 0.01 * np.max(yf)   # 1% of maximum amplitude

        # find first frequency where amplitude < threshold
        idx = np.where(yf < threshold)[0]
        print(xf[idx])

        import matplotlib.pyplot as plt
        plt.plot(xf, yf)
        plt.savefig('Frequency_spectra_cut_for_Gauss_' + str(i))
        plt.close()

        plt.plot(xf,yf)
        plt.xlim(0.5,4.5)
        plt.savefig('Frequency_spectra_cut_for_region_Gauss_' + str(i))
        plt.close()