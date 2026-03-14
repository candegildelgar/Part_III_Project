import xarray as xarr
import numpy as np
import pandas
import matplotlib.pyplot as plt
import os
import obspy.taup.taup_create
import pandas
nc_file= '/raid2/cg812/3d_vp.nc'
stations= pandas.read_csv('/raid2/cg812/2015_stations.csv', header=None)
prem_file= '/raid2/cg812/Velocity_models/gt30km_prem.nd'
densities= pandas.read_csv('/raid2/cg812/Velocity_models/Full_full_PREM.nd', sep= r"\s+", header= None)
print('here')
tom_model= xarr.open_dataset(nc_file, engine="h5netcdf")
print('past this point')
for index, data in stations.iterrows():
        station=data[0]
        lat=data[1]
        lon=data[2]
        velocity = tom_model['dvd'].sel(x=lon,y=lat, method='nearest')
        print(velocity)
        velocity=velocity.sel(depth=slice(0,30))
        deps= list(velocity.depth)
        vs=list(velocity/1.76)
        vp= list(velocity)
        lines_out = []

                                            # Header line for depth=0
                                                
        lines_out.append('{:8.2f}{:12.5f}{:10.5f}{:10.5f}{:10.1f}{:10.1f}'.format(deps[0], vp[0], vs[0], densities.iloc[0,3], 1456, 600))
        for i in range(1, len(deps)):
            lines_out.append('{:8.2f}{:12.5f}{:10.5f}{:10.5f}{:10.1f}{:10.1f}'.format(deps[i], vp[i-1], vs[i-1], densities.iloc[i-1,3], 1456, 600))
            lines_out.append('{:8.2f}{:12.5f}{:10.5f}{:10.5f}{:10.1f}{:10.1f}'.format(deps[i], vp[i], vs[i], densities.iloc[i,3], 1456, 600))
        outdir = '/raid2/cg812/3D_velocity_models'
        output_nd = os.path.join(outdir,  station + '.nd')
        output_npz = os.path.join(outdir, station+ "npz")
        # Concatenate prem.end to the .nd file before TauP conversion
        with open(output_nd, 'w') as f:
            for line in lines_out:
                f.write(str(line) + '\n')
        with open(output_nd, 'a') as out_f, open(prem_file, 'r') as prem_f:
             out_f.write(prem_f.read())
        
        obspy.taup.taup_create.build_taup_model(output_nd, output_folder=outdir)
                                                                                                                                                   





