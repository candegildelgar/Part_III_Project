import pygmt
import obspy
import numpy as np
import glob

plotted_earthquakes = pygmt.Figure()


direc= glob.glob('/raid2/cg812/All_together/*')
#direc.extend(glob.glob('/raid2/cg812/2015_RFs/*'))
plotted_earthquakes.basemap(region="g", projection="N0/15c", frame=True)
plotted_earthquakes.coast(land="#666666", water="skyblue")
ev_lo_list= list()
ev_la_list= list()
for gauss in direc:
    for st in glob.glob(gauss + '/*[!.png]'):
#for station in direc:
#    for st in glob.glob(station +'/*[!.png]'):
        seis= obspy.read(st)
        evlo= seis[0].stats.evlo
        evla= seis[0].stats.evla
        if not evlo in ev_lo_list:
            if not evla in ev_la_list:
                plotted_earthquakes.plot(
                    x=evlo,
                    y=evla, style="p0.1c", color='red')
                ev_lo_list.append(evlo)
                ev_la_list.append(evla)

plotted_earthquakes.savefig('/raid2/cg812/All_RF_earthquakes.png')
