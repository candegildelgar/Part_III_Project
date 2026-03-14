import os
import obspy
from seispy.io import Query
from obspy import UTCDateTime
import glob
import array
from seispy.decon import RFTrace
from obspy.taup import TauPyModel
from obspy.core.inventory import read_inventory
import numpy as np
import matplotlib.pyplot as plt
from obspy.core.trace import Trace
from joblib import parallel_config
from joblib import Parallel, delayed

events=[]
model = TauPyModel(model='prem')

f0 = [1]
itmax = 400
minderr = 0.001
shift= 10
hello= True

fitmin = 0.60  # Minimum 60 percent of radial compoment should be fit (after reconvolving the RF with the vertical component
noisebefore = 0.4  # There should be no peaks before the main P wave which are more than 40% of the amplitude of the Pwave
noiseafter = 0.7  # There should be no peaks after the main P wave which are more than 70% of the amplitude of the Pwave
minamp = 0.04  # There should be signals after the P wave which are at least 4% 

directory= ['/raid2/cg812/Good_2015_earthquakes/VIFE', '/raid2/cg812/Good_2015_earthquakes/LOGR', '/raid2/cg812/Good_2015_earthquakes/NAUG', '/raid2/cg812/Good_2015_earthquakes/DREK', '/raid2/cg812/Good_2015_earthquakes/HOTT', '/raid2/cg812/Good_2015_earthquakes/DYSA', '/raid2/cg812/Good_2015_earthquakes/STOR', '/raid2/cg812/Good_2015_earthquakes/VIKS']
def calculate_RF(station):
    rfstack=list()
    events= glob.glob(station + '/*[!.png]')
        for event in events:
        st= obspy.read(event)
        filenameextract= os.path.basename(event)
        #already= os.path.join('/raid2/cg812/Good_earthquakes/', st[0].stats.station + '/' + filenameextract)
        #if not os.path.exists(already):
        for i in f0:
            direc= '/raid2/cg812/2015_RFs/Gauss_' + str(i) + '.0/' + st[0].stats.station
            if not os.path.exists(direc):
                    os.makedirs(direc)
            rf = RFTrace.deconvolute(st.select(channel='**R')[0], st.select(channel='**Z')[0], method='iter',
                            tshift=shift, f0 = i, itmax = 400, minderr = 0.001)

            #shift it so all peaks align, not necessarily 10 as in the seispy code
            indm = np.argmax(np.abs(rf.data))
            rf.trim(rf.stats.starttime + indm/20 - 5, rf.stats.starttime + indm/20 + 60)
        #wlevel=0.1

            fit= 1- rf.stats.rms[-1]
        #print(fit)
            indm = np.argmax(np.abs(rf.data)) 

    #Select reciever functions with good 
            withinrange = True if (indm > 60  and indm < 140) else False #to convert to seconds times by 20. That should be the direct P arrival, between +2 seconds or -2 seconds (with tshift being 10s, though I could make it longer?)
            if withinrange:
                if fit > fitmin:
                    rf.stats['Ptime']= st[0].stats.Ptime
                    rf.stats['fit']= fit
                    rf.stats['baz']= st[0].stats.baz
                    rf.stats['evla'] = st[0].stats.evla
                    rf.stats['evlo'] = st[0].stats.evlo
                    rf.stats['evdp'] = st[0].stats.evdp
                    rf.stats['stla'] = st[0].stats.stla
                    rf.stats['stlo'] = st[0].stats.stlo
                    rf.stats['dist'] = st[0].stats.dist
                    rf.stats['az'] = st[0].stats.az
                    rf.stats['station'] = st[0].stats.station
                    rf.stats['network'] = st[0].stats.network
                    rf.stats['event'] = st[0].stats.event
                    savepath= os.path.join(direc, st[0].stats.starttime.strftime("%Y%m%dT%H%M%S"))
                    rf.write(savepath, format= 'PICKLE')
                    rf.plot(outfile=savepath)
                else:
                    print('Fit not large enough')

            else:
                print('The main P arrival is not the main phase')


with parallel_config(backend= 'loky', n_jobs=1, verbose=5):
    Parallel()(delayed(calculate_RF)(station) for station in directory)




#Find the fit from the reciever functions?
## Reconvolve with vertical component
    #component2= np.real(st.select(channel='HHR')[0].data)
    #component1= np.real(st.select(channel='HHZ')[0].data)

    #decon= rf.data * np.sum(component1**2)
    #conv=np.real(np.convolve(decon, component1, 'full'))
    #conv=conv[0:len(component2)]
    


    

#Should I also save the P time so i can use it later in Sanne's filtering code?
   
                        
