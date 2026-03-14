import matplotlib.pyplot as plt
import glob
import numpy as np
import obspy
stations= ['VITI', 'FJAS']
done_already=[]
events=[]
for gauss in glob.glob('/raid2/cg812/Refined_automatically/Gauss_*/FJAS'):
        for bin in glob.glob(gauss + '/*'):
            for data in glob.glob(bin + '/*[!.png]'):
                st= obspy.read(data)[0]
                ev=st.stats.event
                events.append(ev)


plt.figure()
sta_average=[]
all_amps= []
in_viti=[]
traces= glob.glob('/raid2/cg812/Good_earthquakes_again/VITI' + '/*[!.png]')
for data in traces:
    st= obspy.read(data)
    if st[0].stats.event in events:
        identifier= str(st[0].stats.starttime) + '_' + 'VITI'
        if identifier not in done_already:
            in_viti.append(st[0].stats.event)
            transverse= st.select(channel='**R')[0].data
            amp= transverse/1000
            all_amps.append(amp)
print(len(all_amps))
for i in range(len(all_amps)):
    tr=all_amps[i]              
    t= np.linspace(0, len(tr), len(tr))
    plt.plot(t, i*50 + tr, color='black')
    done_already.append(identifier)
plt.savefig('/raid2/cg812/RadialVITI' + '.png')

plt.figure()
sta_average=[]
all_amps= []
traces= glob.glob('/raid2/cg812/Good_earthquakes_again/FJAS' + '/*[!.png]')
for data in traces:
    st= obspy.read(data)
    if st[0].stats.event in in_viti:
        identifier= str(st[0].stats.starttime) + '_' + 'FJAS'
        if identifier not in done_already:
            transverse= st.select(channel='**R')[0].data
            amp= transverse/1000
            all_amps.append(amp)
print(len(all_amps))
for i in range(len(all_amps)):
    tr=all_amps[i]              
    t= np.linspace(0, len(tr), len(tr))
    plt.plot(t, i*50 + tr, color='black')
    done_already.append(identifier)
plt.savefig('/raid2/cg812/RadialFJAS' + '.png')