import pygmt
import rasterio
import rioxarray
import geopandas as gpd
gpd.options.io_engine = "fiona"

def dms2dd(degrees, minutes, seconds, direction):
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)
    if direction == 'W' or direction == 'S':
        dd *= -1
    return dd

lonmin= dms2dd(16, 55, 0, 'W')
lonmax= dms2dd(16, 35, 0, 'W')
latmin= dms2dd(64, 57, 0, 'N')
print(latmin)
latmax= dms2dd(65, 7, 0, 'N')
print(latmax)
disp_data_reflate= '/raid2/cg812/disp_2021-2023_mm.tif'
da = rioxarray.open_rasterio(disp_data_reflate).squeeze() * 5

reflation= pygmt.Figure()
reflation.basemap(region=[lonmin, lonmax, latmin, latmax], projection="M4i", frame=True)
reflation.grdimage(
    grid=da, region=[lonmin, lonmax, latmin, latmax],
    projection='M4i',
    frame=True, cmap='viridis'
    )

Calderas=gpd.read_file('/raid2/cg812/Calderas.geojson')
if "index" in Calderas.columns:
    Calderas = Calderas.drop(columns="index")
Calderas=Calderas.to_crs("EPSG:4326")
Askja_lake= gpd.read_file('/raid2/cg812/Askja_lake.geojson')
print(Calderas.crs)
print(Calderas.total_bounds)
reflation.plot(data=Calderas, pen="1p,black")
reflation.plot(data=Askja_lake, pen="1p,black")
reflation.colorbar(frame=["x+lAbsolute ground movement (mm)"])
reflation.savefig('/raid2/cg812/INSAR_data_inflation.png')

