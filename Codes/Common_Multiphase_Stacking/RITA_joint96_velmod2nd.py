
"""
RITA_joint96_velmod2nd.py

Reads a Joint96 model file and prints or writes it in TauP .nd format.
Usage: python RITA_joint96_velmod2nd.py <model_file> [output_nd]
"""


import obspy.taup.taup_create
import os
import pandas
from scipy.interpolate import interp1d


import numpy as np
prem_file= '/raid2/cg812/Velocity_models/gt30km_prem.nd'
    
    
Vp_vals = [np.float64(5.8), np.float64(5.84167), np.float64(4.967497300000001), np.float64(4.967497300000001), np.float64(5.925), np.float64(5.96667), np.float64(6.00833), np.float64(5.1025), np.float64(5.1025), np.float64(6.09167), np.float64(6.13333), np.float64(6.175), np.float64(6.21667), np.float64(6.25833), np.float64(6.3), np.float64(6.34167), np.float64(6.38333), np.float64(6.425), np.float64(6.46667), np.float64(6.50833), np.float64(6.55), np.float64(6.59167), np.float64(6.63333), np.float64(6.675), np.float64(6.71667), np.float64(6.75833), np.float64(8.02138), np.float64(8.02025), np.float64(8.01911), np.float64(8.01798), np.float64(8.01685), np.float64(8.01571)]
Vs_vals = [np.float64(3.2), np.float64(3.22917), np.float64(2.6392473), np.float64(2.6392473), np.float64(3.2875), np.float64(3.31667), np.float64(3.34583), np.float64(2.73375), np.float64(2.73375), np.float64(3.40417), np.float64(3.43333), np.float64(3.4625), np.float64(3.49167), np.float64(3.52083), np.float64(3.55), np.float64(3.57917), np.float64(3.60833), np.float64(3.6375), np.float64(3.66667), np.float64(3.69583), np.float64(3.725), np.float64(3.75417), np.float64(3.78333), np.float64(3.8125), np.float64(3.84167), np.float64(3.87083), np.float64(4.39616), np.float64(4.39639), np.float64(4.39662), np.float64(4.39685), np.float64(4.39708), np.float64(4.39731)]
rho_vals = ([np.float64(2600.0)/1000, np.float64(2612.5)/1000, np.float64(2605.0)/1000, np.float64(2605.0)/1000, np.float64(2637.5)/1000, np.float64(2650.0)/1000, np.float64(2662.5)/1000, np.float64(2655.0)/1000, np.float64(2655.0)/1000, np.float64(2687.5)/1000, np.float64(2700.0)/1000, np.float64(2712.5)/1000, np.float64(2725.0)/1000, np.float64(2737.5)/1000, np.float64(2750.0)/1000, np.float64(2762.5)/1000, np.float64(2775.0)/1000, np.float64(2787.5)/1000, np.float64(2800.0)/1000, np.float64(2812.5)/1000, np.float64(2825.0)/1000, np.float64(2837.5)/1000, np.float64(2850.0)/1000, np.float64(2862.5)/1000, np.float64(2875.0)/1000, np.float64(2887.5)/1000, np.float64(3380.68)/1000, np.float64(3380.58)/1000, np.float64(3380.47)/1000, np.float64(3380.36)/1000, np.float64(3380.25)/1000, np.float64(3380.14)/1000])
deps = [0,1,2, 2.5, 3, 4, 5, 6, 6.5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

print(len(deps))
print(len(Vp_vals))


    # Build depth array using cumulative sum of layer thicknesses
    
    


max_depth = int(np.ceil(deps[-1]))
new_deps = np.arange(0, max_depth + 1, 1)
Vp_vals = np.interp(new_deps, deps, Vp_vals)
Vs_vals = np.interp(new_deps, deps, Vs_vals)
rho = np.interp(new_deps, deps, rho_vals)
deps= new_deps

lines_out = []
# Header line for depth=0
lines_out.append('{:8.2f}{:12.5f}{:10.5f}{:10.5f}{:10.1f}{:10.1f}'.format(deps[0], Vp_vals[0], Vs_vals[0], rho_vals[0], 1456, 600))

moho_dep= 40
for i in range(1, len(deps)):
    # Mark Moho if depth crosses Moho
    if abs(deps[i] - moho_dep) < 1:
        lines_out.append('{:8.2f}{:12.5f}{:10.5f}{:10.5f}{:10.1f}{:10.1f}'.format(deps[i], Vp_vals[i-1], Vs_vals[i-1], rho_vals[i-1], 1456, 600))
        lines_out.append('mantle')
        lines_out.append('{:8.2f}{:12.5f}{:10.5f}{:10.5f}{:10.1f}{:10.1f}'.format(deps[i], Vp_vals[i], Vs_vals[i], rho_vals[i], 1456, 600))
    else:
        lines_out.append('{:8.2f}{:12.5f}{:10.5f}{:10.5f}{:10.1f}{:10.1f}'.format(deps[i], Vp_vals[i-1], Vs_vals[i-1], rho_vals[i-1], 1456, 600))
        lines_out.append('{:8.2f}{:12.5f}{:10.5f}{:10.5f}{:10.1f}{:10.1f}'.format(deps[i], Vp_vals[i], Vs_vals[i], rho_vals[i], 1456, 600))
    

outdir = '/raid2/cg812/Velocity_models'
output_nd = os.path.join(outdir,  "0.1_melt_lenses.nd")
output_npz = os.path.join(outdir, "0.1_melt_lenses.npz")

with open(output_nd, 'w') as f:
    for line in lines_out:
        f.write(str(line) + '\n')


# Concatenate prem.end to the .nd file before TauP conversion

with open(output_nd, 'a') as out_f, open(prem_file, 'r') as prem_f:
    out_f.write(prem_f.read())

# Build TauP .npz model from .nd file
obspy.taup.taup_create.build_taup_model(output_nd, output_folder=outdir)

