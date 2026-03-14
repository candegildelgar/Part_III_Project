import glob
import obspy
import numpy as np
import os
from scipy import stats
lines_out = []

for gauss in glob.glob('/raid2/cg812/Transverse_RFs/*'):
    f0= os.path.basename(gauss)
    print('in ' + f0)
    outside_caldera=[]
    inside_caldera=[]
    for station in glob.glob(gauss + '/*'):
        station_data=[]
        for bin in glob.glob(station + '/*[!.png]'):
            for trace in glob.glob(bin + '/*[!.png]'):
                data= obspy.read(trace)
                x= data[0].data
                sta_id=data[0].stats.station
                square_sum= np.sum(x**2)
                station_data.append(square_sum)
        station_average= np.mean(station_data)
        station_std= np.std(station_data)
        lines_out.append(('with anomalies',f0,sta_id,station_average,station_std))
        z_scores = np.abs(stats.zscore(station_data))
        print(len(z_scores))
        anomalies = np.where(z_scores > 3)
        print(anomalies)
        if len(anomalies[0])>0:
            for i in anomalies[0]:
                del station_data[int(i)]
        print(sta_id)
        print("Anomalies detected at indices (Z-score):", anomalies)
        station_average_without_anomalies= np.mean(station_data)   #not including the anomalous points
        station_std_without_anomalies= np.std(station_data)
        print(station_average_without_anomalies)
        print(station_std_without_anomalies)
        lines_out.append(('without anomalies',f0,sta_id,station_average_without_anomalies,station_std_without_anomalies))
        if sta_id=='APAL' or sta_id=='OLGR' or sta_id== 'MYNG' or sta_id=='HELL' or sta_id== 'OSVA' or sta_id=='VITI' or sta_id == 'OSKV' or sta_id=='ATOP':
            inside_caldera.append(station_average_without_anomalies)
        else:
            outside_caldera.append(station_average_without_anomalies)
    average_inside_caldera= np.sum(inside_caldera)/len(inside_caldera)
    average_outside_caldera= np.sum(outside_caldera)/len(outside_caldera)
    std_inside_caldera=np.std(inside_caldera)
    std_outside_caldera=np.std(outside_caldera)
    print(average_inside_caldera)
    print(average_outside_caldera)
    lines_out.append((f0,'average_inside_caldera',average_inside_caldera, std_inside_caldera))
    lines_out.append((f0,'average_outside_caldera',average_outside_caldera, std_outside_caldera))

outdir = '/raid2/cg812'
savepath = os.path.join(outdir,  "Transverse_power.csv")

with open(savepath, 'w') as f:
    for line in lines_out:
        f.write(str(line) + '\n')


