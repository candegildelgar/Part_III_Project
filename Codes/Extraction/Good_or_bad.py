import os
import obspy
from seispy.io import Query
from obspy import UTCDateTime
import glob
import array
from seispy.decon import RFTrace
from obspy.taup import TauPyModel
from seispy.eq import snr
import numpy as np
from obspy.core.inventory import read_inventory

noiselen= 50


directory= ['/raid2/cg812/2015_earthquakes/VIFE', '/raid2/cg812/2015_earthquakes/TROL', '/raid2/cg812/2015_earthquakes/LOGR', '/raid2/cg812/2015_earthquakes/NAUG', '/raid2/cg812/2015_earthquakes/DREK', '/raid2/cg812/2015_earthquakes/DYFE', '/raid2/cg812/2015_earthquakes/HOTT', '/raid2/cg812/2015_earthquakes/DYSA', '/raid2/cg812/2015_earthquakes/STOR', '/raid2/cg812/2015_earthquakes/VIKS']
model = TauPyModel(model='iasp91')
shift = 10
time_after = 120
from joblib import parallel_config
from joblib import Parallel, delayed

def good_or_bad(station):
    events=[]
    print('in' + station)
    data= glob.glob(station + '/*')
    for event in data:
        filenameextract= os.path.basename(event)
        date_str = filenameextract[:15]
        if date_str not in events:
            events.append(date_str)

    for date in events:
        
        print('reading date' + date)
        response= False
        path= os.path.join(station, f"{date}.*.PICKLE")
        st = obspy.read(path)

        #Processing the data
        #Already filtered for it during extraction data
        st.detrend('linear') #detrend
        

        #For applying a cosine taper could use obspy.simulate_seismometer? Would check with Tom
        st.taper(max_percentage= 0.05, type='cosine')

        #For ensuring they are the same length
        start = max(tr.stats.starttime for tr in st)
        end   = min(tr.stats.endtime for tr in st)
        st.trim(start, end)

        st.interpolate(sampling_rate=20)
        st._cleanup()

        #filter data
        st.filter('highpass', freq= 0.05)

        #for tr in st:
            #Remove instrument response- I actually got the wrong network code for the tom winder file- should work until 2025
            #xml_file= read_inventory('/raid4/tpo21/scripts/tom_mega_comb_april_24.xml')
            #channel_map = {
                #"*HN": "HHN",
                #"*HE": "HHE",
                #"*HZ":  "HHZ",
                #"HHN": "HHN",
                #"HHE": "HHE",
                #"HHZ":  "HHZ",
                #"N2": "HHN",
                #"E2": "HHE",
                #"Z2":  "HHZ", # if you have vertical channels
            #}
            #tr.stats.network= '8K'
            #component_code= tr.stats.channel

           
            #try:
                #tr.stats.channel = channel_map[component_code]
                #try:
                    #tr.remove_response(xml_file)
                    #print('response removed')
                
                #except:
                    #channel_map = {
                    #"HHN": "BHN",
                    #"HHE": "BHE",
                    #"HHZ":  "BHZ"
                    # if you have vertical channels
                #}
                    #tr.stats.channel = channel_map[tr.stats.channel]
                    #try:
                        #tr.remove_response(xml_file)
                        #print('response removed')

                    #except:
                        #channel_map = {
                    #"BHN": "LHN",
                    #"BHE": "LHE",
                    #"BHZ":  "LHZ"
                    #}    
                        #tr.stats.channel = channel_map[tr.stats.channel]
                        #try:
                            #tr.remove_response(xml_file)
                        
                        #except:
                            #print('couldnt remove response')
                            #print('try different xml file')
                            #xml_file=read_inventory('/raid2/cg812/combined_dataless', format= 'STATIONXML')
                            #channel_map = {
                                #"LHN": "HHN",
                                #"LHE": "HHE",
                                #"LHZ":  "HHZ"}
                           
                            #try:
                                #tr.stats.channel = channel_map[tr.stats.channel]
                                #try:
                                    #tr.remove_response(xml_file)
                                    #print('response removed')
                                
                                #except:
                                    #channel_map = {
                                    #"HHN": "BHN",
                                    #"HHE": "BHE",
                                    #"HHZ":  "BHZ"
                                    # if you have vertical channels
                                #}
                                    #tr.stats.channel = channel_map[tr.stats.channel]
                                    #try:
                                        #tr.remove_response(xml_file)
                                        #print('response removed')

                                    #except:
                                        #channel_map = {
                                    #"BHN": "LHN",
                                    #"BHE": "LHE",
                                    #"BHZ":  "LHZ"
                                    #}   
                                        #tr.stats.channel = channel_map[tr.stats.channel]    
                                        #try:
                                            #tr.remove_response(xml_file)
                                        
                                        #except:
                                            #print('give up')
                                            #response=False

                            #except:
                                #print('different channel code')
            #except:
                #print('different channel code')
                #response= False
            
            


        try:
            st_TRZ = st.copy().rotate('NE->RT', back_azimuth= st[0].stats['baz'])
        
            # Check rotation worked
            rotated_channels = [tr.stats.channel for tr in st_TRZ]
            if any(ch.endswith('R') for ch in rotated_channels):

                arrivals = model.get_travel_times(st_TRZ[0].stats['evdp'], st_TRZ[0].stats['dist'], phase_list=['P'])
                P_arr = arrivals[0]

                st_TRZ[0].stats['evla'] = st[0].stats.evla
                st_TRZ[0].stats['evlo'] = st[0].stats.evlo
                st_TRZ[0].stats['evdp'] = st[0].stats.evdp
                st_TRZ[0].stats['stla'] = st[0].stats.stla
                st_TRZ[0].stats['stlo'] = st[0].stats.stlo
                st_TRZ[0].stats['dist'] = st[0].stats.dist
                st_TRZ[0].stats['az'] = st[0].stats.az
                st_TRZ[0].stats['baz'] = st[0].stats.baz
                st_TRZ[0].stats['station'] = st[0].stats.station
                st_TRZ[0].stats['network'] = st[0].stats.network
                st_TRZ[0].stats['event'] = st[0].stats.event
                st_TRZ[0].stats['Ptime']= P_arr
                
                for tr in st_TRZ:
                        # Make copies to trim
                    sig = tr.copy().trim(tr.stats.starttime + P_arr.time,
                                            tr.stats.starttime + P_arr.time + noiselen)
                    noise = tr.copy().trim(tr.stats.starttime + P_arr.time - noiselen,
                                            tr.stats.starttime + P_arr.time)
                        # seispy.eq.snr expects (signal, noise) for one component
                    
                    tr.stats['snr'] =snr(sig.data, noise.data)   

                st_TRZ.trim(st_TRZ[0].stats.starttime+P_arr.time-shift,
                st_TRZ[0].stats.starttime+P_arr.time+time_after)

                if len(st_TRZ) == 3:
                    if st_TRZ[0].stats['snr']<1 or st_TRZ[1].stats['snr']<1 or st_TRZ[2].stats['snr']<1:
                        print('Signal too small')

                    else:
                        print('saving')
                
                        time_axis = st_TRZ[0].times() - shift
                        direc= os.path.join('/raid2/cg812/Good_2015_earthquakes/', st[0].stats.station)
                        if not os.path.exists(direc):
                            os.makedirs(direc)
                        savepath= os.path.join(direc, st[0].stats.starttime.strftime("%Y%m%dT%H%M%S") + f"processed")
                        st_TRZ.plot(outfile= savepath)
                        st_TRZ.write(savepath, format='PICKLE')
                        
                else:
                    print('skipping because it only has 2 components')
                
            else:
                    print('couldnt rotate')
        except:
                print('couldnt rotate')

    print('Finished')


with parallel_config(backend= 'loky', n_jobs=2, verbose=5):
    Parallel()(delayed(good_or_bad)(station) for station in directory)

