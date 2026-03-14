import glob
import os
import numpy as np
direc= glob.glob('/raid2/cg812/Raw_data/*')
direc.extend(glob.glob('/raid2/cg812/2015_earthquakes/*'))
lines_out=[]
inside_caldera=list()
outside_caldera=list()

for station in direc:
    sta_id= os.path.basename(station)
    n_raw_earthquakes= len(glob.glob(station + '/*'))/3
    n_good_earthquakes= len(glob.glob('/raid2/cg812/Migrated_to_depth_3D/1/' + sta_id + '_*'))
    percentage= n_good_earthquakes/n_raw_earthquakes *100
    lines_out.append((sta_id, percentage, 'N/A'))
    if percentage >0:
    #if sta_id != 'DREK' and sta_id != 'DYSA' and sta_id != 'HOTT' and sta_id != 'LOGR' and sta_id != 'NAUG' and sta_id != 'STOR' and sta_id != 'VIFE':
        if sta_id=='APAL' or sta_id=='OLGR' or sta_id== 'MYNG' or sta_id=='HELL' or sta_id== 'OSVA' or sta_id=='VITI' or sta_id == 'OSKV' or sta_id=='ATOP' or sta_id=='JONG' or sta_id=='JONS' or sta_id=='GODA' or sta_id=='VBOR' or sta_id=='KLUR':
            inside_caldera.append(percentage)
        else:
            outside_caldera.append(percentage)

average_inside_caldera= np.sum(inside_caldera)/len(inside_caldera)
std_inside_caldera=np.std(inside_caldera)
average_outside_caldera= np.sum(outside_caldera)/len(outside_caldera)
std_outside_caldera=np.std(outside_caldera)
print(average_inside_caldera)
print(std_inside_caldera)
print(average_outside_caldera)
print(std_outside_caldera)
lines_out.append(('average_inside_caldera',average_inside_caldera, std_inside_caldera))
lines_out.append(('average_outside_caldera',average_outside_caldera, std_outside_caldera))

outdir = '/raid2/cg812'
savepath = os.path.join(outdir,  "Percentage_good.csv")

with open(savepath, 'w') as f:
    for line in lines_out:
        f.write(str(line) + '\n')


