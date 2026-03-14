import os
import glob
import obspy
import subprocess
from obspy.taup import TauPyModel

depth= 10
phase= 'P'
mod=TauPyModel(model='/raid2/cg812/Velocity_models/icelandic_crust_with_0.1_melt.npz')
lonpp = []
latpp = []
lonsta= []
latsta= []

#PP= open('PP_' + str(depth) + 'km_' + str(phase) + '.txt', 'w')
direc=['/raid2/cg812/Synthetic_RF_with_two_chambers/Gauss_2_0.1_melt_icelandic_crust.pkl']* len(glob.glob('/raid2/cg812/All_together/Gauss_2.0/*[!.png]'))
trace_list= glob.glob('/raid2/cg812/All_together/Gauss_2.0/*[!.png]')
for count in range(len(direc)):
    station_current= trace_list[count]
    try:
        current_trace= obspy.read(station_current)
    except:
        print('theres gonna be an extra one here')
    BAZ= current_trace[0].stats.baz
    sta_id = current_trace[0].stats.station
    stat_lat = current_trace[0].stats.stla
    stat_lon = current_trace[0].stats.stlo
    seis= obspy.read(direc[count])
    test = [
            '/raid2/cg812/TauP-3.1.0/bin/taup pierce --mod ' + str(mod) + ' -h ' + str(
                current_trace[0].stats.evdp) + ' -p ' + phase + ' --pierce ' + str(depth) + ' --nodiscon --sta ' + str(current_trace[0].stats.stla) + ' ' + str(current_trace[0].stats.stlo) + ' --evt ' + str(current_trace[0].stats.evla) + ' ' + str(current_trace[0].stats.evlo)]
    
    
    # Run test in terminal
    out = subprocess.check_output(
        test,
        shell=True,
        universal_newlines=True)

        # Split the output into lines
        # t[0] is a description
        # t[1] the downwards pierce and t[2] the upwards pierce (if event depth < PP depth)
        # t[1] the upwards pierce (if event depth > PP depth)
    t = out.split('\n')

    # Split the relevant line into strings
    if current_trace[0].stats.evdp <= float(depth):
        u = t[2].split()
    else:
        u = t[1].split()
    
    # For the string U: PP depth = u[1], lat = u[3], lon = u[4]
    lonpp.append(float(u[4]))
    latpp.append(float(u[3]))

plotted_piercepoints= pygmt.Figure()
topo_data= '/raid5/Iceland/data/IslandsDEM/IslandsDEMv0_10x10m_zmasl_isn2016_PROJECTED.grd'



# Set bounds of map based on piercepoints
lonmin = np.min(lonpp) - 0.5
lonmax = np.max(lonpp) + 0.5
latmin = np.min(latpp) - 0.5
latmax = np.max(latpp) + 0.5

pygmt.makecpt(
    cmap='gray', series='100/1600',
    continuous=True
)
plotted_piercepoints.grdimage(
    grid=topo_data, region=[lonmin, lonmax, latmin, latmax],
    projection='M4i',
    frame=True
    )
# llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
# are the lat/lon values of the lower left and upper right corners
# of the map.
# resolution = 'i' means use intermediate resolution coastlines.
# lon_0, lat_0 are the central longitude and latitude of the projection.
#m = Basemap(
    #llcrnrlon=lonmin, llcrnrlat=latmin, urcrnrlon=lonmax, urcrnrlat=latmax,
      #      resolution='i', projection='M4i', lon_0=np.mean((lonmin, lonmax)), lat_0=np.mean((latmin, latmax)), epsg=	4326)
#m.drawcoastlines()
#m.drawcountries()

# draw parallels and meridians.
#m.drawparallels(np.arange(-40, 80., 5.), color='gray')
#m.drawmeridians(np.arange(-30., 80., 5.), color='gray')
#m.arcgisimage( server='https://services.arcgisonline.com/ArcGIS',
    #service='World_Topo_Map', xpixels = 2000, verbose= True)


# plot pierce points
if "index" in H.columns:
    H = H.drop(columns="index")
if "index" in K.columns:
    K = K.drop(columns="index")
if "index" in V.columns:
    V = V.drop(columns="index")
if "index" in lake.columns:
    lake = lake.drop(columns="index")
if "index" in calderas.columns:
    calderas = calderas.drop(columns="index")

plotted_piercepoints.plot(data= lake, fill= 'lightblue')
plotted_piercepoints.plot(data= H)
plotted_piercepoints.plot(data= K)
plotted_piercepoints.plot(data= V)
plotted_piercepoints.plot(data= calderas)
#x1, y1 = m(lonpp, latpp)
#m.scatter(x1, y1, s=10, marker='o', color='k', alpha=.3)

plotted_piercepoints.plot(x=lonpp, y=latpp, style= 'c0.1c', fill='red3')


#x2, y2= m(lonsta, latsta)
#m.scatter(x2, y2, s=10, marker= 'o', color= 'r')

plotted_piercepoints.savefig('/raid2/cg812/PP_map_10km_only_caldera_stations_synthetic.pdf')
