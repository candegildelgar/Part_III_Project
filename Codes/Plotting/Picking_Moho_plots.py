import pygmt
import matplotlib.pyplot as plt
import pandas
from geopy.distance import geodesic
import numpy as np
import geopandas as gpd
gpd.options.io_engine = "fiona"

letters=['A','B','C','D','E']
lettersNS=['B','A','C','F','D', 'E']
EW_all_lat=list()
EW_all_lon=list()
EW_all_CPP= list()
NS_all_lat=list()
NS_all_lon=list()
NS_all_CPP= list()
all_lat=list()
all_lon=list()
all_CPP= list()

for i in range(6):
    cross_section= pandas.read_csv('/raid2/cg812/Grids/cross-section_NS_' + str(lettersNS[i]) + '.lonlat', sep= r"\s+", header= None)
    CPP= pandas.read_csv('/raid2/cg812/Picking_Moho_only_multiples/NS_' + str(lettersNS[i]) + '_only_multiples_CPP_40.csv')
    grid_dist = np.arange(len(cross_section.iloc[:,0])) * 1

    for i in range(len(CPP)): 
        if CPP.iloc[i,7] == 'Y':
            NS_all_CPP.append([CPP.iloc[i,6],cross_section.iloc[i,1], cross_section.iloc[i,0]])
        if CPP.iloc[i,7] == 'M':
            NS_all_CPP.append([0,cross_section.iloc[i,1], cross_section.iloc[i,0]])
            
    


for i in range(0,5):
    cross_section= pandas.read_csv('/raid2/cg812/Grids/cross-section_EW_' + str(letters[i]) + '.lonlat', sep= r"\s+", header= None)
    CPP= pandas.read_csv('/raid2/cg812/Picking_Moho_only_multiples/EW_' + str(letters[i]) + '_only_multiples_CPP_40.csv')
    grid_dist = np.arange(len(cross_section.iloc[:,0])) * 1
    for i in range(len(CPP)): 
        if CPP.iloc[i,7] == 'Y':
            EW_all_CPP.append([CPP.iloc[i,6],cross_section.iloc[i,1], cross_section.iloc[i,0]])
        if CPP.iloc[i,7] == 'M':
            EW_all_CPP.append([0,cross_section.iloc[i,1], cross_section.iloc[i,0]])
    
minlon, maxlon = -17.14, -16.14
minlat, maxlat = 64.8, 65.2
all_lat.extend(EW_all_lat)
all_lat.extend(NS_all_lat)
all_lon.extend(EW_all_lon)
all_lon.extend(NS_all_lon)
all_CPP.extend(EW_all_CPP)
all_CPP.extend(NS_all_CPP)

fig= pygmt.Figure()

fig.basemap(region=[minlon, maxlon, minlat, maxlat], projection="M4i", frame=True)
pygmt.makecpt(
    cmap="viridis", series=[np.min([row[0] for row in all_CPP]), np.max([row[0] for row in all_CPP])], reverse=True

)


fig.plot(
    x=[row[2] for row in all_CPP],
    y=[row[1] for row in all_CPP],
    fill=[row[0] for row in all_CPP],
    cmap=True,
    style="c0.3c",
    pen="black",
)
fig.colorbar(frame="xaf+lDepth (km)")
fig.savefig('/raid2/cg812/40km_plots.png')

all_CPP_mesh_grid= pygmt.blockmean(x=[row[2] for row in all_CPP], y=[row[1] for row in all_CPP], z=[row[0] for row in all_CPP], region= [minlon, maxlon, minlat, maxlat], spacing= '1m')
Combined_moho_grid= pygmt.surface(data= all_CPP_mesh_grid, region= [minlon, maxlon, minlat, maxlat], spacing= '1m')
Moho= pygmt.Figure()
pygmt.makecpt(
    cmap="viridis", series=[np.min([row[0] for row in all_CPP]), np.max([row[0] for row in all_CPP])], reverse=True

)
Moho.grdimage(
    grid=Combined_moho_grid, region= [minlon, maxlon, minlat, maxlat],
    projection='M4i',
    frame=True, cmap=True
    )

H=gpd.read_file('/raid2/cg812/Herðubreið.geojson')
K= gpd.read_file('/raid2/cg812/Kollóttadyngja.geojson')
V= gpd.read_file('/raid2/cg812/Vadala.geojson')
Askja_lake=gpd.read_file('/raid2/cg812/Askja_lake.geojson')
Calderas=gpd.read_file('/raid2/cg812/Calderas.geojson')
VZ=gpd.read_file('/raid2/cg812/Volcanic_zones1.geojson')

if "index" in H.columns:
    H = H.drop(columns="index")
if "index" in K.columns:
    K = K.drop(columns="index")
if "index" in V.columns:
    V = V.drop(columns="index")
if "index" in Askja_lake.columns:
    Askja_lake = Askja_lake.drop(columns="index")

if "index" in Calderas.columns:
    Calderas = Calderas.drop(columns="index")

if "index" in VZ.columns:
    VZ = VZ.drop(columns="index")

Moho.plot(data=H)
Moho.plot(data=K)
Moho.plot(data=V)
Moho.plot(data=VZ)


Moho.colorbar(frame="xaf+lDepth (km)")
Moho.show()
Moho.savefig('/raid2/cg812/40km_using_all_CPP.png')


cross_section= pandas.read_csv('/raid2/cg812/Grids/cross-section_EW_C.lonlat', sep= r"\s+", header= None)
CPP= pandas.read_csv('/raid2/cg812/Picking_4_6km/EW_C_CPP_6.csv')
fig, ax= plt.subplots()
ax.set_ylim(0,15)
grid_dist = np.arange(len(cross_section.iloc[:,0])) * 1
for item in range(len(grid_dist)):
    if CPP.iloc[item,7]== 'Y':
        ax.scatter(CPP.iloc[item,5], CPP.iloc[item,6], color= 'blue')
        print(CPP.iloc[item,5])
plt.xlabel('Horizontal distance (km)')
plt.ylabel('Depth(km)')
ax.invert_yaxis()
fig.savefig('EW_C_6.png')

