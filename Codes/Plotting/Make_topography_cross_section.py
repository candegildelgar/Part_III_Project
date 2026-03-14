import pygmt
import geopandas as gpd
import pandas as pd
gpd.options.io_engine= "fiona"
fig = pygmt.Figure()

topo_data= pygmt.datasets.load_earth_relief(resolution="15s", region=[-17.5, -16, 64.6, 65.4])
latA, lonA = 65.2, -16.74
latB, lonB = 64.8, -16.74
track_df = pygmt.project(
    center=[lonA, latA],  # Start point of survey line (longitude, latitude)
    endpoint=[lonB, latB],  # End point of survey line (longitude, latitude)
    generate=0.1, unit=True  # Output data in steps of 0.1 degrees
)
track_df = pygmt.grdtrack(grid=topo_data, points=track_df, newcolname="elevation")
region = [
    0,
    track_df.p.max()+0.01,
    0,
    track_df.elevation.max()
]

fig.basemap(
    region=region,
    projection="X12c/6c",
    frame=["xaf", "yaf", "+tEW Profile"],
)
fig.plot(
    x=track_df.p,
    y=track_df.elevation,
    fill="gray",
    pen="1p,black,solid",
    close="+y-8000",
)


fig.plot(x=65.025, y=0, style="t0.3c")
fig.show()
fig.savefig('NS_C_topography.png')
                         