import xarray as xarr
import numpy as np
import matplotlib.pyplot as plt
import pygmt
import geopandas as gpd
import pandas
gpd.options.io_engine = "fiona"

nc_file= '/home/tpo21/Public/candela/3d_vp.nc'
tom_model= xarr.open_dataset(nc_file)





npoints = 50  # resolution along section
lons = np.linspace(-17.14, -16.14, npoints)
lats = [65.175]*npoints
section=tom_model['dvd'].sel(depth=slice(0, 15)).interp(y=("points", lats),
    x=("points", lons))
plot_1_d= tom_model['dvd'].sel(x=-16.34, y=65.025, method='nearest').sel(depth=slice(0,15))
plot_1_d.plot(y="depth")
plt.gca().invert_yaxis()
plt.savefig('1d_model_Vp.png')
plt.close()

plot_1_d_average= tom_model['dvd'].sel(x=slice(-17.14, -16.14), y=slice(64.8,65.2),depth=slice(0,30)).mean(dim=['x', 'y'])
plot_1_d_average.plot(y="depth")
plt.gca().invert_yaxis()
plt.savefig('1d_average_model_Vp.png')

plt.figure()
stations= pandas.read_csv('/raid2/cg812/Stations_to_use for_real.csv', header=None)
for index, data in stations.iterrows():
        station=data[0]
        lat=data[1]
        lon=data[2]
        velocity = tom_model['dvd'].sel(x=lon,y=lat, method='nearest').sel(depth=slice(0,30))
        velocity.plot(y='depth')

plt.gca().invert_yaxis()
plt.savefig('stations_model_Vp.png')

vs_model= pandas.read_csv('/raid2/cg812/Velocity_models/rob_Vs_1D_for_candela.txt', sep= r"\s+")
plt.figure()
plt.plot(vs_model.iloc[:,1]*1.76, vs_model.iloc[:,0])
plt.gca().invert_yaxis()
plt.savefig('surface_wave_model_Vp.png')


plt.figure(figsize=(8,6))
plot_2_d_slice= tom_model['dvd'].sel(x=slice(-17.14,-16.14), y=slice(64.8, 65.2)).sel(depth=4)- 5.8


plot_2_d_slice
twod_slice= pygmt.Figure()
pygmt.makecpt(
    cmap="matplotlib/plasma", series=[plot_2_d_slice.min().values,plot_2_d_slice.max().values], reverse=True

)

twod_slice.grdimage(
    grid=plot_2_d_slice, region= [-17.14, -16.14, 64.8, 65.2],
    projection='M4i',
    frame=True, cmap=True
    )

VZ=gpd.read_file('/raid2/cg812/Volcanic_zones1.geojson')
if "index" in VZ.columns:
    VZ = VZ.drop(columns="index")

twod_slice.plot(data=VZ)
twod_slice.colorbar(frame="xaf+lDifference in Velocity (km/s) with respect to AK135")
twod_slice.show()
twod_slice.savefig('/raid2/cg812/2d_slice_Vp_model_4.png')
