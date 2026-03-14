from matplotlib import pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=(10, 8))
vs=[3.20000, 3.87083, 4.39616, 4.39616]
depth=[0.5, 23.5, 24.5, 35]

ax.plot(vs, depth, color="black", linewidth=0.8)
ax.invert_yaxis()
ax.set_ylabel('Depth(km)')
ax.set_xlabel('Vs velocity (km/s)')
plt.savefig('/raid2/cg812/Smooth_crust_plot.png')