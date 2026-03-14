from matplotlib import pyplot as plt
import obspy
from obspy import Trace
import glob
import numpy as np
import os

directories= ['/raid2/cg812/Transverse_RFs/Gauss_6.0']
for gauss in directories:
    for station in glob.glob(gauss + '/*'):
        print('in ' + station)
        avs_data=list()
        avs_baz=list()
        data= list()
        bins= glob.glob(station + '/*[!.png]')
        for azimuth in bins:
            bin_dat=list()
            baz=list()
            rf= glob.glob(azimuth + '/*[!.png]')
            for st in rf:
                tr= obspy.read(st)
                bin_dat.append(tr[0].data)
                baz.append(tr[0].stats.baz)
                tr.trim(tr[0].stats.starttime, tr[0].stats.starttime +35)
            average_rf= np.mean(np.vstack(bin_dat), axis=0)
            avs_baz.append(np.mean(baz))
            amp= average_rf/np.max(np.abs(average_rf))
            #amp= average_rf*10
            avs_data.append(amp)

        overall_average= np.mean(np.vstack(bin_dat), axis=0)
        overall_average= overall_average/np.max(np.abs(overall_average))

    
        savepath= os.path.join(station, 'back_azimuth_plot_averaged.png')
        fig, ax = plt.subplots()
        for i in range(len(avs_data)):
            t = np.linspace(0, 65, len(avs_data[i]))
            ax.plot(t, avs_data[i]*50 + avs_baz[i], color='k', linewidth=0.5)
            
            plt.fill_between(t, avs_baz[i], avs_data[i]*50 + avs_baz[i], where= avs_data[i]*50 + avs_baz[i]>avs_baz[i], color='red')
            plt.fill_between(t, avs_baz[i], avs_data[i]*50 + avs_baz[i], where= avs_data[i]*50 + avs_baz[i]<avs_baz[i], color='blue')
        ax.set_title("Average RF")
        ax.set_xlabel('Time (s)')
        ax.set_ylim(-40,400)
        #ax.set_ylabel('Back azimuth (degrees)')
        ax.set_ylabel('Amplitude')
        fig.savefig(savepath)
        plt.close(fig)

        savepath= os.path.join(station, 'back_azimuth_plot_average_of_all.png')
        fig, ax = plt.subplots()
        t = np.linspace(0, 35, len(overall_average))
        ax.plot(t, overall_average, color='k', linewidth=0.5)
        plt.fill_between(t, 0, overall_average, where= overall_average>0, color='red')
        plt.fill_between(t, 0, overall_average, where= overall_average<0, color='blue')
        ax.set_title("Average RF")
        ax.set_xlabel('Time (s)')
        #ax.set_ylabel('Back azimuth (degrees)')
        ax.set_ylabel('Amplitude')
        #fig.savefig(savepath)
        plt.close(fig)
    
    
