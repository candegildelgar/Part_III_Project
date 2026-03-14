import obspy
from obspy.clients.fdsn import Client as IRISClient
from obspy import UTCDateTime
import os.path
import obspy.geodetics.base
import numpy as np
import obspy.geodetics
import glob
import os
import sys
import re
from obspy.core.inventory import read_inventory
import pandas as pd
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel
from seispy.eq import snr
from joblib import parallel_config
from joblib import Parallel, delayed
              
xml_file= read_inventory('/raid4/tpo21/scripts/tom_mega_comb_april_24.xml')


directories= ['/raid5/Iceland/data/ARCHIVE/2015','/raid5/Iceland/data/ARCHIVE/2016','/raid5/Iceland/data/ARCHIVE/2017', '/raid5/Iceland/data/ARCHIVE/2018', '/raid5/Iceland/data/ARCHIVE/2019', '/raid5/Iceland/data/ARCHIVE/2020', '/raid5/Iceland/data/ARCHIVE/2021'] 
irisclient = IRISClient("IRIS")
dataclient = IRISClient("IRIS")
#do the same thing for the 2024 and 2025 data. 

def Extract_data_and_refine(directory):
#define the parameters for the events
    radmin = 30  # Minimum radius for teleseismic earthquakes
    radmax = 90  # Maximum radius (further distances interact with the core-mantle boundary
    minmag = 5.5  # Minumum magnitude of quake
    maxmag = 10  # Maximum magnitude of quake
    lengthoftrace = 30. * 60.  # 30 min
    max_freq= 2
    min_freq= 0.05
    event_count= 0
    pattern = r'^[^_]+_[^_]+_([^_]+)_([^_]+)\.m$' #to extract the name
    stations_to_use= pd.read_csv('/raid2/cg812/Stations_to_use.csv', header=None)

    
    for i in directory:
        print(f"\n📁 Searching in: {directory}")
        for moment in  glob.glob(i + '/*'):
            day= glob.glob(moment + '/*')
            for filename in day:
                sta= obspy.read(filename)
                filenameextract= os.path.basename(filename)
                match= re.search(pattern, filenameextract)
                sta_code= match.group(1)  #to extract the name I want (the station code)
                match_exists = (stations_to_use == sta_code).any().any()
                if match_exists:
                    direc= '/raid2/cg812/Raw_data/' + sta_code 
                    if not os.path.exists(direc):
                        os.makedirs(direc)
                    
                    #Merge to get rid of the sampling gaps
                    sta.merge(method=1, fill_value=0)
                    sta.resample(20)
                    
                    #has most of the stuff in it, not all (see word doc for ones missing)
                    selected_inventory = xml_file.select(station=sta_code)
                    if selected_inventory:
                        station_metadata = selected_inventory[0].stations[0]
                
                    else:
                        print(f"Station {sta_code} not found in inventory, skipping.")
                
                    row = stations_to_use[stations_to_use.iloc[:, 0] == sta_code]
                    sta_latitude= row.iloc[0,1]
                    sta_longitude= row.iloc[0,2]
                    mintime= sta[0].stats.starttime
                    maxtime= sta[0].stats.endtime
                
            
                    try:
                        cat = irisclient.get_events(
                        latitude= sta_latitude,
                        longitude= sta_longitude,
                        minradius=radmin,
                        maxradius=radmax,
                        starttime=mintime,
                        endtime= maxtime,
                        minmagnitude=minmag, maxmagnitude=maxmag)

                    
                        for ev in cat:
                            print('found event')
                            event_count= event_count + 1
                            evtime= ev.origins[0].time
                            t = UTCDateTime(evtime)#so that it can handle it
                            
                            try:
                                seis = sta.copy().trim(evtime, evtime + lengthoftrace)
                            except Exception as e:
                                print(f"Trim failed for {sta_code}: {e}")
                                continue
                    
                            filename = os.path.join(direc, seis[0].stats.starttime.strftime("%Y%m%dT%H%M%S") + f".{seis[0].stats.channel}.PICKLE")
                            if os.path.exists(filename):
                                print("Already processed:", filename)
                                continue
                            
                              # try:
                               #     seis.remove_response(xml_file)
                               # except:
                                #    print('Couldnt remove response')
                                 #   direc= '/raid2/cg812/2015_No_response/' + sta_code 
                                  #  if not os.path.exists(direc):
                                   #     os.makedirs(direc)
                                    #filename = os.path.join(direc, seis[0].stats.starttime.strftime("%Y%m%dT%H%M%S") + f".{seis[0].stats.channel}.PICKLE")
                                    #continue
                            
                            
                            else:   
                                evtlatitude= ev.origins[0]['latitude']
                                evtlongitude = ev.origins[0]['longitude']
                                try:
                                    evtdepth = ev.origins[0][
                                    'depth'] / 1.e3  # convert to km from m
                                except:
                                    print('failed to get true depth')
                                    evtdepth = 30.
                    # Compute distances azimuth and backazimuth, can't do it yet as missing the data
                                print('getting distances')
                            #somehow after here it fails, not sure why
                                distm, az, baz = obspy.geodetics.base.gps2dist_azimuth(
                                    float(evtlatitude), float(evtlongitude), float(sta_latitude), float(sta_longitude))
                                distdg = distm / (6371.e3 * np.pi / 180.)

                    
                    #Put in the dictionary a bunch of things
                                print('putting stuff in the dictionary')
                                seis[0].stats['evla'] = evtlatitude
                                seis[0].stats['evlo'] = evtlongitude
                                seis[0].stats['evdp'] = evtdepth
                                seis[0].stats['stla'] = sta_latitude
                                seis[0].stats['stlo'] = sta_longitude
                                seis[0].stats['dist'] = distdg
                                seis[0].stats['az'] = az
                                seis[0].stats['baz'] = baz
                                seis[0].stats['event'] = ev
                                try:
                                    channel_inv= selected_inventory.select(channel= seis[0].stats.channel)
                                    try:
                                        seis[0].stats['azimuth']= channel_inv[0].stations[0].channels[0].azimuth 
                                        seis[0].stats['dip']= channel_inv[0].stations[0].channels[0].dip
                                    except Exception as e:
                                        print('failed to get azimuth or dip')
                                        print("Error:", repr(e))
                                except:
                                    print('No match found in xml')
                                                    
                                
                    # Write out to filest
                                try:
                                    seis.write(filename, format='PICKLE') #python object hierarchy
                                    print('Writing out filename ' + filename)
                                except Exception as e:
                                    print("Error:", repr(e))
                        
                    except:
                        do_nothing= 'do nothing'

            #if len(stack)>0:
                #fig, ax = plt.subplots(figsize=(10, 8))
                #scale = 0.5          # amplitude scaling
            # offset_step = 1.0    # vertical spacing between traces
                #for i, tr in enumerate(stack):
                    #data = tr.data.astype(float)
                    #data /= np.max(np.abs(data))  # normalize per trace

                    #npts = tr.stats.npts
                    #dt = tr.stats.delta
                    #starttime = tr.stats.starttime

                    # Build actual time vector (in seconds relative to start of record)
                    #time = np.arange(npts) * dt
                    # Or absolute times if you want timestamps:
                    # time = np.array([starttime + j * dt for j in range(npts)])

            #        Vertical offset to stack traces
                    #offset = i * offset_step

                    #ax.plot(time, data * scale + offset, color="black", linewidth=0.8)


                    # Optional: Label each trace using station/channel info
                    #ax.text(time[-1] + 0.1, offset, f"{tr.stats.station}.{tr.stats.channel}",
                    #va='center', fontsize=8)
            # ax.set_xlabel("Time (s)")
            # ax.set_ylabel("Trace Index (stacked vertically)")
                #ax.set_title("Vertically Stacked Seismic Wiggles (MiniSEED)")
                #ax.set_ylim(-offset_step, len(stack)*offset_step)
                #ax.invert_yaxis()  # top trace first
                #ax.grid(True, alpha=0.3) 
                #starttime = stack[0].stats.starttime
                #safe_dir = directory.replace('/', '_').replace('\\', '_')

                # Format timestamp and combine safely
                #safe_time = f"{starttime.strftime('%Y%m%dT%H%M%S')}_{safe_dir}"

                # Define final save path
                #save_path = os.path.join('/raid2/cg812/Stacks/', f"{safe_time}_stack.png")
                
                #plt.savefig(save_path)


    print('Finished')

Extract_data_and_refine(directory=directories)
