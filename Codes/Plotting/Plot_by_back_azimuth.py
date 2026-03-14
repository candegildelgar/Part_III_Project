from matplotlib import pyplot as plt
import obspy
import glob
import numpy as np
import os

directories= ['/raid2/cg812/Processed_RF/Gauss_1.0/*']
for direc in directories:
    directory= glob.glob(direc)
    for station in directory:
        fig, ax = plt.subplots()
        bins= glob.glob(station + '/*[!.png]')
        for azimuth in bins:
            rf= glob.glob(azimuth + '/*[!.png]')
            for st in rf:
                tr= obspy.read(st)
                t= np.linspace(0, tr[0].stats.npts / tr[0].stats.sampling_rate, tr[0].stats.npts)
                ax.plot(t, tr[0].data/np.max(tr[0].data) + tr[0].stats.baz/10, color='k', linewidth=0.5)
                ax.set_title("RF by back azimuth Gauss")
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Back azimuth')
                ax.text(t[-1] + 0.1, tr[0].stats.baz/10, f"{tr[0].stats.starttime}",
                        va='center', fontsize=8)
            
        savepath= os.path.join(station, 'back_azimuth_plot')
        fig.savefig(savepath)


