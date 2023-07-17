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
from subgrid_radiation import auxiliary
from subgrid_radiation.sun_position_array import horizon
from subgrid_radiation import sun_position_array

mpl.style.use("classic")

# %matplotlib auto
# %matplotlib auto

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Grid for lookup positions
# lu_lon = np.linspace(-180.0, 170.0, 36, dtype=np.float64)  # 10 degree
# lu_lat = np.linspace(0.0, 90.0, 10, dtype=np.float64)  # 10 degree
lu_lon = np.linspace(-180.0, 175.0, 72, dtype=np.float64)  # 5 degree
lu_lat = np.linspace(0.0, 90.0, 19, dtype=np.float64)  # 5 degree

# Ray-tracing and 'SW_dir_cor' calculation
dist_search = 100.0  # search distance for terrain shading [kilometre]
geom_type = "grid"  # "grid" or "quad"
sw_dir_cor_max = 25.0
ang_max = 89.9
# hori_azim_num, hori_acc = 30, 3.0
# hori_azim_num, hori_acc = 45, 2.0
# hori_azim_num, hori_acc = 60, 1.5
hori_azim_num, hori_acc = 72, 1.25
# hori_azim_num, hori_acc = 180, 0.5
# hori_azim_num, hori_acc = 360, 0.25
ray_algorithm = "guess_constant"
elev_ang_low_lim = -85.0  # -15.0

# Miscellaneous settings
path_work = "/Users/csteger/Desktop/dir_work/"  # working directory
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
# z[z > 0.0] *= (-1)  # create depression
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
vert_grid = auxiliary.rearrange_pad_buffer(x, y, z)
print("Size of elevation data: %.3f" % (vert_grid.nbytes / (10 ** 9))
      + " GB")

# Merge vertex coordinates and pad geometry buffer (0.0 m elevation)
slice_in = (slice(pixel_per_gc * offset_gc, -pixel_per_gc * offset_gc),
            slice(pixel_per_gc * offset_gc, -pixel_per_gc * offset_gc))
z_zero = np.zeros_like(z)
dem_dim_in_0, dem_dim_in_1 = z_zero[slice_in].shape
vert_grid_in = auxiliary.rearrange_pad_buffer(x[slice_in], y[slice_in],
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
                          z_sun[:, :, np.newaxis]), axis=2)

# Mask (optional)
num_gc_y = int((x.shape[0] - 1) / pixel_per_gc) - 2 * offset_gc
num_gc_x = int((x.shape[1] - 1) / pixel_per_gc) - 2 * offset_gc
mask = np.ones((num_gc_y, num_gc_x), dtype=np.uint8)
# mask[:] = 0
# mask[20:40, 20:40] = 1

# -----------------------------------------------------------------------------
# Compute spatially aggregated correction factors
# -----------------------------------------------------------------------------

# Compute sky view factor
north_pole = np.array([0.0, 10.0 ** 10, 0.0], dtype=np.float64)
sky_view_factor, area_increase_factor, sky_view_area_factor, slope, aspect \
    = sun_position_array.horizon.sky_view_factor(
        vert_grid, dem_dim_0, dem_dim_1,
        vert_grid_in, dem_dim_in_0, dem_dim_in_1,
        north_pole, pixel_per_gc, offset_gc,
        mask=mask, dist_search=dist_search, hori_azim_num=hori_azim_num,
        hori_acc=hori_acc, ray_algorithm=ray_algorithm,
        elev_ang_low_lim=elev_ang_low_lim, geom_type=geom_type)

# Compute sky view factor and SW_dir correction factor
sw_dir_cor, sky_view_factor_2nd, area_increase_factor_2nd, \
    sky_view_area_factor_2nd, slope_2nd, aspect_2nd \
    = sun_position_array.horizon.sky_view_factor_sw_dir_cor(
        vert_grid, dem_dim_0, dem_dim_1,
        vert_grid_in, dem_dim_in_0, dem_dim_in_1,
        north_pole, sun_pos, pixel_per_gc, offset_gc,
        mask=mask, dist_search=dist_search, hori_azim_num=hori_azim_num,
        hori_acc=hori_acc, ray_algorithm=ray_algorithm,
        elev_ang_low_lim=elev_ang_low_lim, geom_type=geom_type,
        ang_max=ang_max, sw_dir_cor_max=sw_dir_cor_max)

# Check equlity of output
print(np.abs(sky_view_factor - sky_view_factor_2nd).max())
print(np.abs(area_increase_factor - area_increase_factor_2nd).max())
print(np.abs(sky_view_area_factor - sky_view_area_factor_2nd).max())
print(np.abs(slope - slope_2nd).max())
print(np.abs(aspect - aspect_2nd).max())

# Test plot for 2D-fields
data_2d = {"sky_view_factor": sky_view_factor,
            "area_increase_factor": area_increase_factor,
            "sky_view_area_factor": sky_view_area_factor,
           "surface_slope": slope,
           "surface_aspect": aspect}
for i in data_2d.keys():
    plt.figure()
    plt.pcolormesh(data_2d[i])
    plt.colorbar()
    plt.title(i.replace("_", " "), fontweight="bold",
              loc="left", y=1.01, fontsize=13)
    txt = "(min/max: %.3f" % np.nanmin(data_2d[i]) \
          + ", %.3f" % np.nanmax(data_2d[i]) + ")"
    plt.title(txt, loc="right", y=1.01, fontsize=12)
print("Spatial mean of sky view area factor: %.2f"
      % np.nanmean(data_2d["sky_view_area_factor"]))

# Test plot for f_cor
if plot:
    levels = np.arange(0.0, 2.0, 0.2)
    cmap = cm.roma_r
    norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False,
                                   extend="max")
    plt.figure()
    ind_2, ind_3 = 5, 0  # 3, 0
    plt.pcolormesh(sw_dir_cor[:, :, ind_2, ind_3], cmap=cmap, norm=norm)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.colorbar()
    plt.title("Sun position (lat/lon): %.2f" % lu_lat[ind_2]
              + ", %.2f" % lu_lon[ind_3] + " [degree]",
              fontsize=12, fontweight="bold", y=1.01)
    print("Spatially averaged 'sw_dir_cor': %.5f"
          % np.nanmean(sw_dir_cor[:, :, ind_2, ind_3]))
print("Range of 'sw_dir_cor'-values: [%.2f" % np.nanmin(sw_dir_cor)
      + ", %.2f" % np.nanmax(sw_dir_cor) + "]")

# Compare with results from 'rays'-method
sw_dir_cor_rays = np.load(path_work + "SW_dir_cor_artifical_rays.npy")

ind_2, ind_3 = 5, 0  # 3, 0
plt.figure()
plt.pcolormesh(sw_dir_cor[:, :, ind_2, ind_3]
               - sw_dir_cor_rays[:, :, ind_2, ind_3])
plt.colorbar()
plt.title("hori_azim_num: " + str(hori_azim_num), fontsize=12, y=1.01)

# Deviation statistics
print(np.abs(sw_dir_cor - sw_dir_cor_rays).max())
print(np.percentile(np.abs(sw_dir_cor - sw_dir_cor_rays), 99.5))
print(np.abs(sw_dir_cor - sw_dir_cor_rays).mean())
