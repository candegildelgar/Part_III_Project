import glob
import os
import numpy as np
import obspy
direc= glob.glob('/raid2/cg812/Migrated_to_depth_3D/*')
events=[]
for gauss in direc:
    for data in glob.glob(gauss + '/*[!.png]'):
        st= obspy.read(data)[0]
        ev=st.stats.event
        if ev not in events:
            events.append(ev)

print(len(events))