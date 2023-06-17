# Description: Test computation of subgrid correction factors for direct
#              downward shortwave radiation (aggregated to the model grid cell
#              resolution) for an array of sun positions. Use an artifical
#              terrain and a local coordinate system.

# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cmcrameri import cm
import subgrid_radiation as subrad

mpl.style.use("classic")

# %matplotlib auto
# %matplotlib auto

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Grid for lookup positions
lu_lon = np.linspace(-180.0, 170.0, 36, dtype=np.float64)  # 10 degree
lu_lat = np.linspace(0.0, 90.0, 10, dtype=np.float64)  # 10 degree

# Ray-tracing and 'SW_dir_cor' calculation
dist_search = 100.0  # search distance for terrain shading [kilometre]
geom_type = "grid"  # "grid" or "quad"
ang_max = 89.5
sw_dir_cor_max = 20.0

# Miscellaneous settings
dir_work = "/Users/csteger/Desktop/dir_work/"  # working directory
plot = True

# -----------------------------------------------------------------------------
# Generate artifical data and check
# -----------------------------------------------------------------------------

# Coordinates
x = np.linspace(-50000, 50000, 1001, dtype=np.float32)
y = np.linspace(-50000, 50000, 1001, dtype=np.float32)
x, y = np.meshgrid(x, y)

# 'Lowered' Hemisphere (-> avoid very steep sides at base)
radius = 20000.0
lower = 5000.0
z = np.sqrt(radius ** 2 - x ** 2 - y ** 2) - lower
z[np.isnan(z)] = 0.0
z[z < 0.0] = 0.0
print("Elevation (min/max): %.2f" % z.min() + ", %.2f" % z.max())

# # Gaussian mountain
# amp = 15000.0  # amplitude
# sigma = 10000.0
# z = amp * np.exp(-(x ** 2 / (2.0 * sigma ** 2)
#                    + y ** 2 / (2.0 * sigma ** 2)))
# print("Elevation (min/max): %.2f" % z.min() + ", %.2f" % z.max())

# Test plot
if plot:
    plt.figure()
    plt.pcolormesh(x, y, z)
    plt.colorbar()
    plt.figure(figsize=(12, 3))
    plt.plot(x[500, :], z[500, :])

pixel_per_gc = 10
offset_gc = 10

# Merge vertex coordinates and pad geometry buffer
dem_dim_0, dem_dim_1 = x.shape
vert_grid = subrad.auxiliary.rearrange_pad_buffer(x, y, z)
print("Size of elevation data: %.3f" % (vert_grid.nbytes / (10 ** 9))
      + " GB")

# Merge vertex coordinates and pad geometry buffer (0.0 m elevation)
slice_in = (slice(pixel_per_gc * offset_gc, -pixel_per_gc * offset_gc),
            slice(pixel_per_gc * offset_gc, -pixel_per_gc * offset_gc))
z_zero = np.zeros_like(z)
dem_dim_in_0, dem_dim_in_1 = z_zero[slice_in].shape
vert_grid_in = subrad.auxiliary.rearrange_pad_buffer(x[slice_in], y[slice_in],
                                                   z_zero[slice_in])
print("Size of elevation data (0.0 m surface): %.3f"
      % (vert_grid_in.nbytes / (10 ** 9)) + " GB")

# Sun positions
r_sun = 20000.0 + 10.0 ** 9
lu_lon_2d, lu_lat_2d = np.meshgrid(lu_lon, lu_lat)
x_sun = r_sun * np.cos(np.deg2rad(lu_lat_2d)) * np.cos(np.deg2rad(lu_lon_2d))
y_sun = r_sun * np.cos(np.deg2rad(lu_lat_2d)) * np.sin(np.deg2rad(lu_lon_2d))
z_sun = r_sun * np.sin(np.deg2rad(lu_lat_2d))
sun_pos = np.concatenate((x_sun[:, :, np.newaxis],
                          y_sun[:, :, np.newaxis],
                          z_sun[:, :, np.newaxis]), axis=2).astype(np.float32)

# -----------------------------------------------------------------------------
# Compute spatially aggregated correction factors
# -----------------------------------------------------------------------------

# Compute
sw_dir_cor = subrad.subsolar_lookup.sw_dir_cor_coherent_rp8(
    vert_grid, dem_dim_0, dem_dim_1,
    vert_grid_in, dem_dim_in_0, dem_dim_in_1,
    sun_pos, pixel_per_gc, offset_gc,
    dist_search=dist_search, geom_type=geom_type,
    ang_max=ang_max, sw_dir_cor_max=sw_dir_cor_max)

# Check output
print("Range of 'sw_dir_cor'-values: [%.2f" % sw_dir_cor.min()
      + ", %.2f" % sw_dir_cor.max() + "]")

# Test plot
if plot:
    levels = np.arange(0.0, 2.0, 0.2)
    cmap = cm.roma_r
    norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False,
                                   extend="max")
    plt.figure()
    ind_2, ind_3 = 3, 0  # 3, 0
    plt.pcolormesh(sw_dir_cor[:, :, ind_2, ind_3], cmap=cmap, norm=norm)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.colorbar()
    plt.title("Sun position (lat/lon): %.2f" % lu_lat[ind_2]
              + ", %.2f" % lu_lon[ind_3] + " [degree]",
              fontsize=12, fontweight="bold", y=1.01)
    print("Spatially averaged 'sw_dir_cor': %.5f"
          % sw_dir_cor[:, :, ind_2, ind_3].mean())
