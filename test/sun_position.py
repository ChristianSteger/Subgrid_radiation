# Description: Test computation of subgrid correction factors for direct
#              downward shortwave radiation (aggregated to the model grid cell
#              resolution) for a single sun position
#
# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib as mpl
from skyfield.api import Distance
from cmcrameri import cm
from netCDF4 import Dataset
import time
from subgrid_radiation import transform, auxiliary
from subgrid_radiation import sun_position
from utilities.grid import grid_frame
from utilities.plot import truncate_colormap

mpl.style.use("classic")

# %matplotlib auto
# %matplotlib auto

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Ray-tracing and 'SW_dir_cor' calculation
dist_search = 100.0  # search distance for terrain shading [kilometre]
geom_type = "grid"  # "grid" or "quad"
ang_max = 89.5
sw_dir_cor_max = 20.0

# Miscellaneous settings
path_work = "/Users/csteger/Desktop/dir_work/"  # working directory
plot = True

# -----------------------------------------------------------------------------
# Load and check data
# -----------------------------------------------------------------------------

# Load data
ds = xr.open_dataset(path_work + "MERIT_remapped_COSMO_0.02deg.nc")
pixel_per_gc = ds.attrs["pixels_per_grid_cell_zonal"]
# pixel per grid cell (along one dimension)
offset_gc = ds.attrs["offset_grid_cells_zonal"]
# offset in number of grid cells
# -----------------------------------------------------------------------------
# sub-domain with reduces boundary: 30 x 50
offset_gc = int(offset_gc / 2)
ds = ds.isel(rlat=slice(325 * pixel_per_gc - pixel_per_gc * offset_gc,
                        355 * pixel_per_gc + 1 + pixel_per_gc * offset_gc),
             rlon=slice(265 * pixel_per_gc - pixel_per_gc * offset_gc,
                        315 * pixel_per_gc + 1 + pixel_per_gc * offset_gc))
# -----------------------------------------------------------------------------
# # sub-domain: 390 x 490
# ds = ds.isel(rlat=slice(100 * pixel_per_gc - pixel_per_gc * offset_gc,
#                         490 * pixel_per_gc + 1 + pixel_per_gc * offset_gc),
#              rlon=slice(100 * pixel_per_gc - pixel_per_gc * offset_gc,
#                         590 * pixel_per_gc + 1 + pixel_per_gc * offset_gc))
# -----------------------------------------------------------------------------
# # sub-domain: 240 x 290
# ds = ds.isel(rlat=slice(200 * pixel_per_gc - pixel_per_gc * offset_gc,
#                         440 * pixel_per_gc + 1 + pixel_per_gc * offset_gc),
#              rlon=slice(200 * pixel_per_gc - pixel_per_gc * offset_gc,
#                         490 * pixel_per_gc + 1 + pixel_per_gc * offset_gc))
# -----------------------------------------------------------------------------
lon = ds["lon"].values.astype(np.float64)
lat = ds["lat"].values.astype(np.float64)
elevation = ds["Elevation"].values
pole_lon = ds["rotated_pole"].grid_north_pole_longitude
pole_lat = ds["rotated_pole"].grid_north_pole_latitude
rlon = ds["rlon"].values
rlat = ds["rlat"].values
ds.close()

# Test plot
if plot:
    cmap = truncate_colormap(plt.get_cmap("terrain"), val_min=0.3, val_max=1.0)
    levels = np.arange(0.0, 5000.0, 250.0)
    norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="max")
    # -------------------------------------------------------------------------
    ccrs_rot_pole = ccrs.RotatedPole(pole_latitude=pole_lat,
                                     pole_longitude=pole_lon)
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs_rot_pole)
    plt.pcolormesh(rlon, rlat, elevation, cmap=cmap, norm=norm)
    # -------------------------------------------------------------------------
    plt.vlines(x=rlon[0:None:pixel_per_gc], ymin=rlat[0], ymax=rlat[-1],
               color="grey", lw=0.5, zorder=2)
    plt.hlines(y=rlat[0:None:pixel_per_gc], xmin=rlon[0], xmax=rlon[-1],
               color="grey", lw=0.5, zorder=2)
    # -------------------------------------------------------------------------
    rlon_fr, rlat_fr = grid_frame(rlon, rlat, offset=0)
    poly = plt.Polygon(list(zip(rlon_fr, rlat_fr)), facecolor="none",
                       edgecolor="black", alpha=1.0, linewidth=1.5, zorder=3)
    ax.add_patch(poly)
    rlon_fr, rlat_fr = grid_frame(rlon, rlat, offset=pixel_per_gc * offset_gc)
    poly = plt.Polygon(list(zip(rlon_fr, rlat_fr)), facecolor="none",
                       edgecolor="black", alpha=1.0, linewidth=1.5, zorder=3)
    ax.add_patch(poly)
    # -------------------------------------------------------------------------
    ax.set_aspect("auto")
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linestyle="-", zorder=3)
    plt.colorbar()

# -----------------------------------------------------------------------------
# Coordinate transformation
# -----------------------------------------------------------------------------

# Transform elevation data (geographic/geodetic -> ENU coordinates)
x_ecef, y_ecef, z_ecef = transform.lonlat2ecef(lon, lat, elevation,
                                               ellps="sphere")
dem_dim_0, dem_dim_1 = elevation.shape
trans_ecef2enu = transform.TransformerEcef2enu(
    lon_or=lon.mean(), lat_or=lat.mean(), ellps="sphere")
x_enu, y_enu, z_enu = transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                         trans_ecef2enu)
del x_ecef, y_ecef, z_ecef

# Test plot
if plot:
    plt.figure()
    plt.pcolormesh(x_enu, y_enu, z_enu)
    plt.colorbar()

# Merge vertex coordinates and pad geometry buffer
vert_grid = auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)
print("Size of elevation data: %.3f" % (vert_grid.nbytes / (10 ** 9))
      + " GB")
del x_enu, y_enu, z_enu

# Transform 0.0 m surface data (geographic/geodetic -> ENU coordinates)
slice_in = (slice(pixel_per_gc * offset_gc, -pixel_per_gc * offset_gc),
            slice(pixel_per_gc * offset_gc, -pixel_per_gc * offset_gc))
elevation_zero = np.zeros_like(elevation)
x_ecef, y_ecef, z_ecef \
    = transform.lonlat2ecef(lon[slice_in], lat[slice_in],
                            elevation_zero[slice_in], ellps="sphere")
dem_dim_in_0, dem_dim_in_1 = elevation_zero[slice_in].shape
x_enu, y_enu, z_enu = transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                         trans_ecef2enu)
del x_ecef, y_ecef, z_ecef
del lon, lat, elevation

# Test plot
if plot:
    plt.figure()
    plt.pcolormesh(x_enu, y_enu, z_enu)
    plt.colorbar()

# Merge vertex coordinates and pad geometry buffer
vert_grid_in = auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)
print("Size of elevation data (0.0 m surface): %.3f"
      % (vert_grid_in.nbytes / (10 ** 9)) + " GB")
del x_enu, y_enu, z_enu

# -----------------------------------------------------------------------------
# Compute correction factors for single sun position
# -----------------------------------------------------------------------------

# Initialise terrain
terrain = sun_position.Terrain()
terrain.initialise(
    vert_grid, dem_dim_0, dem_dim_1,
    vert_grid_in, dem_dim_in_0, dem_dim_in_1,
    pixel_per_gc, offset_gc, dist_search,
    geom_type=geom_type, ang_max=ang_max,
    sw_dir_cor_max=sw_dir_cor_max)

# Allocate output array
num_gc_y = int((dem_dim_in_0 - 1) / pixel_per_gc)
num_gc_x = int((dem_dim_in_1 - 1) / pixel_per_gc)
sw_dir_cor = np.empty((num_gc_y, num_gc_x), dtype=np.float32)
sw_dir_cor.fill(0.0) # default value

# Compute f_cor for specific sun position
subsol_lon = np.array([12.0], dtype=np.float64)  # (12.0, 20.0, ...) [degree]
subsol_lat = np.array([-23.5], dtype=np.float64) # (-23.5, 0.0, 23.5) [degree]
subsol_dist = np.empty(subsol_lon.shape, dtype=np.float32)
subsol_dist[:] = Distance(au=1).m
# astronomical unit (~average Sun-Earth distance) [m]
x_ecef, y_ecef, z_ecef \
    = transform.lonlat2ecef(subsol_lon, subsol_lat, subsol_dist,
                            ellps="sphere")
x_enu, y_enu, z_enu = transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                         trans_ecef2enu)
sun_pos = np.array([x_enu[0], y_enu[0], z_enu[0]], dtype=np.float32)
print((" Default: ").center(79, "-"))
terrain.sw_dir_cor(sun_pos, sw_dir_cor)
sw_dir_cor_def = sw_dir_cor.copy()
print((" Coherent rays: ").center(79, "-"))
terrain.sw_dir_cor_coherent(sun_pos, sw_dir_cor)
print("Maximal absolute deviation: %.6f"
      % np.abs(sw_dir_cor - sw_dir_cor_def).max())
print((" Coherent rays (packages with 8 rays): ").center(79, "-"))
terrain.sw_dir_cor_coherent_rp8(sun_pos, sw_dir_cor)
print("Maximal absolute deviation: %.6f"
      % np.abs(sw_dir_cor - sw_dir_cor_def).max())

# Check output
print("Range of 'sw_dir_cor'-values: [%.2f" % sw_dir_cor.min()
      + ", %.2f" % sw_dir_cor.max() + "]")

# Test plot
levels = np.arange(0.0, 2.0, 0.2)
cmap = cm.roma_r
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False,
                               extend="max")
if plot:
    plt.figure()
    plt.pcolormesh(sw_dir_cor, cmap=cmap, norm=norm)
    plt.colorbar()

# -----------------------------------------------------------------------------
# Compute correction factors for an array of sun positions
# -----------------------------------------------------------------------------

# subsol_lon_1d = np.linspace(-180.0, 162.0, 10, dtype=np.float64)  # 38 degree
# subsol_lat_1d = np.linspace(-23.5, 23.5, 5, dtype=np.float64)  # 11.75 degree
subsol_lon_1d = np.linspace(-180.0, 172.0, 45, dtype=np.float64)  # 8 degree
subsol_lat_1d = np.linspace(-23.5, 23.5, 15, dtype=np.float64)  # 3.36 degree
sw_dir_cor_arr = np.empty(sw_dir_cor.shape
                          + (subsol_lat_1d.size, subsol_lon_1d.size),
                          dtype=np.float32)
subsol_dist = np.array([Distance(au=1).m], dtype=np.float32)
t_beg = time.time()
for ind_i, i in enumerate(subsol_lat_1d):
    for ind_j, j in enumerate(subsol_lon_1d):
        x_ecef, y_ecef, z_ecef \
            = transform.lonlat2ecef(np.array([j], dtype=np.float64),
                                    np.array([i], dtype=np.float64),
                                    subsol_dist, ellps="sphere")
        x_enu, y_enu, z_enu = transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                                 trans_ecef2enu)
        sun_pos = np.array([x_enu[0], y_enu[0], z_enu[0]], dtype=np.float32)
        # terrain.sw_dir_cor(sun_pos, sw_dir_cor)
        # terrain.sw_dir_cor_coherent(sun_pos, sw_dir_cor)
        terrain.sw_dir_cor_coherent_rp8(sun_pos, sw_dir_cor)
        sw_dir_cor_arr[:, :, ind_i, ind_j] = sw_dir_cor
print("Elapsed time: " + "%.2f" % (time.time() - t_beg) + " sec")

# Compare result with output from 'subsolar_lookup'
ds = xr.open_dataset(path_work + "SW_dir_cor_lookup.nc")
ind_beg = np.where(subsol_lon_1d == ds["subsolar_lon"].values[0])[0][0]
ind_end = ind_beg + ds["subsolar_lon"].size
sw_dir_cor_lut = ds["f_cor"].values
ds.close()
dev_abs = np.abs(sw_dir_cor_arr[:, :, :, ind_beg:ind_end] - sw_dir_cor_lut)
print("Maximal absolute deviation: %.8f" % dev_abs.max())
print("Mean absolute deviation: %.8f" % dev_abs.mean())
