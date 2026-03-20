import glob
import obspy
import shutil
import os
import numpy as np

directories= glob.glob('/raid2/cg812/Water_level_decon/*')
azimuth_bins=np.linspace(0,360, 37)
for direc in directories:
    directory= glob.glob(direc + '/*')
    for station in directory:
        bin_paths=list()
        for i in range (len(azimuth_bins)-1):
            bin_paths.append(os.path.join(station, 'bin_range' + str(azimuth_bins[i]) + 'to' + str(azimuth_bins[i+1])))
        rf= glob.glob(station + '/[!bin]*[!.png]')
        
        for st in rf:
            tr= obspy.read(st)[0]
            for i in range(37):
                if tr.stats.baz<i*10:
                    
                    if not os.path.exists(bin_paths[i-1]):
                        os.makedirs(bin_paths[i-1])
                    shutil.move(st, bin_paths[i-1])
                    shutil.move(st + '.png', bin_paths[i-1])
                    break

            

