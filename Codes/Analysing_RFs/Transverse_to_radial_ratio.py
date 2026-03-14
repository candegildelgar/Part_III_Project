import glob
import obspy
import os
import numpy as np
from scipy import stats
outside_caldera=[]
inside_caldera=[]
lines_out = []
stations=['APAL','ASK', 'DALR', 'FJAS', 'FLAT', 'FLUR', 'GODA', 'HOTR', 'HRUR', 'JONG', 'JONS', 'KATT', 'KLUR', 'LIND', 'LOKT', 'MIDF', 'MOFO', 'MYNG', 'MYVO', 'OLGR', 'OSVA', 'RIFR', 'RODG', 'SOSU', 'STAM', 'SVAD', 'TOHR', 'TOLI', 'VADA', 'VBOR', 'VITI', 'HELL', 'ATOP', 'OSKV']
stations_2015=['DREK', 'DYSA', 'HOTT', 'LOGR', 'NAUG', 'STOR', 'VIFE', 'VIKS']




for sta_id in stations_2015:
    done_already=[]
    events=[]
    sta_data=[]
    for gauss in glob.glob('/raid2/cg812/2015_RFs/Gauss_*/' + sta_id):
        print(gauss)
        for bin in glob.glob(gauss + '/*'):
            for data in glob.glob(bin + '/*[!.png]'):
                st= obspy.read(data)[0]
                ev=st.stats.event
                events.append(ev)
        traces= glob.glob('/raid2/cg812/Good_2015_earthquakes/' + sta_id + '/*[!.png]')
        for data in traces:
            st= obspy.read(data)
            if st[0].stats.event in events:
                identifier= str(st[0].stats.event) + '_' + sta_id
                if identifier not in done_already:
                    transverse= st.select(channel='**T')[0].data
                    done_already.append(identifier)
                    radial= st.select(channel='**R')[0].data
                    ratio= np.sum(transverse**2)/np.sum(radial**2)
                    sta_data.append(ratio)

    sta_average=np.sum(sta_data)/len(sta_data)
    sta_std=np.std(sta_data)
    lines_out.append(('with anomalies', sta_id, sta_average, sta_std))
    z_scores = np.abs(stats.zscore(sta_data))
    print(len(z_scores))
    anomalies = np.where(z_scores > 3)

    if len(anomalies[0])>0:
        for i in anomalies[0]:
            lines_out.append(('anomalies at ', done_already[int(i)] ))
            del sta_data[int(i)]
    print(sta_id)
    print("Anomalies detected at indices (Z-score):", anomalies)
    station_average_without_anomalies= np.mean(sta_data)   #not including the anomalous points
    station_std_without_anomalies= np.std(sta_data)
    lines_out.append(('with anomalies', sta_id, station_average_without_anomalies, station_std_without_anomalies))
    if sta_id != 'STOR':
        if sta_id=='APAL' or sta_id=='OLGR' or sta_id== 'MYNG' or sta_id=='HELL' or sta_id== 'OSVA' or sta_id=='VITI' or sta_id == 'OSKV' or sta_id=='ATOP' or sta_id=='JONG' or sta_id=='JONS' or sta_id=='GODA' or sta_id=='VBOR' or sta_id=='KLUR':
            inside_caldera.append(station_average_without_anomalies)
        else:
            outside_caldera.append(station_average_without_anomalies)

for sta_id in stations:
    events=[]
    sta_data=[]
    done_already=[]
    for gauss in glob.glob('/raid2/cg812/Refined_automatically/Gauss_*/' + sta_id):
        for bin in glob.glob(gauss + '/*'):
            for data in glob.glob(bin + '/*[!.png]'):
                st= obspy.read(data)[0]
                ev=st.stats.event
                events.append(ev)
                sta_id= st.stats.station
        traces= glob.glob('/raid2/cg812/Good_earthquakes_again/' + sta_id + '/*[!.png]')
        for data in traces:
            st= obspy.read(data)
            if st[0].stats.event in events:
                identifier= str(st[0].stats.starttime) + '_' + sta_id
                if identifier not in done_already:
                    transverse= st.select(channel='**T')[0].data
                    done_already.append(identifier)
                    radial= st.select(channel='**R')[0].data
                    ratio= np.sum(transverse**2)/np.sum(radial**2)
                    sta_data.append(ratio)

            
            
    sta_average=np.sum(sta_data)/len(sta_data)
    sta_std=np.std(sta_data)
    lines_out.append(('with anomalies', sta_id, sta_average, sta_std))
    z_scores = np.abs(stats.zscore(sta_data))
    print(len(z_scores))
    anomalies = np.where(z_scores > 3)

    if len(anomalies[0])>0:
        for i in anomalies[0]:
            lines_out.append(('anomalies at ', done_already[int(i)] ))
            del sta_data[int(i)]
    print(sta_id)
    print("Anomalies detected at indices (Z-score):", anomalies)
    station_average_without_anomalies= np.mean(sta_data)   #not including the anomalous points
    station_std_without_anomalies= np.std(sta_data)
    lines_out.append(('with anomalies', sta_id, station_average_without_anomalies, station_std_without_anomalies))
    if sta_id != 'STOR':
        if sta_id=='APAL' or sta_id=='OLGR' or sta_id== 'MYNG' or sta_id=='HELL' or sta_id== 'OSVA' or sta_id=='VITI' or sta_id == 'OSKV' or sta_id=='ATOP' or sta_id=='JONG' or sta_id=='JONS' or sta_id=='GODA' or sta_id=='VBOR' or sta_id=='KLUR':
            inside_caldera.append(station_average_without_anomalies)
        else:
            outside_caldera.append(station_average_without_anomalies)
    else:
        print('hello')


average_inside_caldera= np.sum(inside_caldera)/len(inside_caldera)
average_outside_caldera= np.sum(outside_caldera)/len(outside_caldera)
std_inside_caldera=np.std(inside_caldera)
std_outside_caldera=np.std(outside_caldera)
lines_out.append(('average_inside_caldera',average_inside_caldera, std_inside_caldera))
print(average_inside_caldera)
print(average_outside_caldera)
lines_out.append(('average_outside_caldera',average_outside_caldera, std_outside_caldera))

outdir = '/raid2/cg812'
savepath = os.path.join(outdir,  "Transverse_to_radial_ratio.csv")

with open(savepath, 'w') as f:
    for line in lines_out:
        f.write(str(line) + '\n')