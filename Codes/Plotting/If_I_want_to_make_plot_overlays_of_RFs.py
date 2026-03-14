import obspy
from matplotlib import pyplot as plt
import numpy as np
data=obspy.read('/raid2/cg812/Migrated_to_depth_3D/1/JONS_2015-05-12T07:16:27.150000Z.pkl')
indm=np.argmax(np.abs(data[0].data))
data.trim(data[0].stats.starttime + indm/20 - 5, data[0].stats.starttime + indm/20 + 60)
#data.write('/raid2/cg812/Synthetic_RF_with_two_chambers/Gauss_2_just_Moho.pkl', format= 'PICKLE')
filter='1'
stats = data[0].stats['conversions'][filter]
RF_depths_PSs, RF_amps_PSs, RF_H_PSs = stats['depth_PSs'], stats['amp_PSs'], stats['Hdist_PSs']
RF_depths_PPs, RF_amps_PPs, RF_H_PPs = stats['depth_PPs'], stats['amp_PPs'], stats['Hdist_PPs']
data=obspy.read('/raid2/cg812/Migrated_to_depth_3D/1/JONS_2015-05-12T07:16:27.150000Z.pkl')
indm=np.argmax(np.abs(data[0].data))
data.trim(data[0].stats.starttime + indm/20 - 5, data[0].stats.starttime + indm/20 + 60)
filter='1'
stats = data[0].stats['conversions'][filter]
RF_depths, RF_amps_Ps, RF_H_Ps = stats['depth_Ps'], stats['amp_Ps'], stats['Hdist_Ps']
plt.figure(figsize=(8, 6))
plt.plot(RF_amps_Ps, RF_depths, 'black', label='Ps')
plt.xlabel('RF amplitude', fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Depth (km)', fontsize=20)
plt.ylim(0,60)
plt.gca().invert_yaxis()
plt.savefig('/raid2/cg812/PS_try_RF')
plt.figure(figsize=(8, 6))
plt.plot(RF_amps_PPs, RF_depths_PPs, 'black', label='Ps')
plt.xlabel('RF amplitude', fontsize=20)
plt.ylabel('Depth (km)', fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0,60)
plt.gca().invert_yaxis()
plt.savefig('/raid2/cg812/PPs_try_RF')
plt.figure(figsize=(8, 6))
plt.plot(RF_amps_PSs, RF_depths_PSs, 'black', label='Ps')
plt.xlabel('RF amplitude', fontsize=20)
plt.ylabel('Depth (km)',fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0,60)
plt.gca().invert_yaxis()
plt.savefig('/raid2/cg812/PSs_try_RF')