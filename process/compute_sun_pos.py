# Description: Compute subgrid correction factors for direct shortwave
#              downward radiation (aggregated to the model grid cell
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
import shortwave_subgrid as swsg
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
geom_type = "grid"  # "quad" ca. 25% faster than "grid"
ang_max = 89.5
sw_dir_cor_max = 20.0

# Miscellaneous settings
dir_work = "/Users/csteger/Desktop/dir_work/"  # working directory
ellps = "sphere"  # Earth's surface approximation (sphere, GRS80 or WGS84)
plot = False

# -----------------------------------------------------------------------------
# Load and check data
# -----------------------------------------------------------------------------

# Load data
ds = xr.open_dataset(dir_work + "MERIT_remapped_COSMO.nc")
pixel_per_gc = ds.attrs["sub_grid_info_zonal"]
# pixel per grid cell (along one dimension)
offset_gc = ds.attrs["offset_grid_cells_zonal"]
# offset in number of grid cells
# -----------------------------------------------------------------------------
# # -> use sub-domain with reduces boundary for now...
# offset_gc = int(offset_gc / 2)
# ds = ds.isel(rlat=slice(325 * pixel_per_gc - pixel_per_gc * offset_gc,
#                         355 * pixel_per_gc + 1 + pixel_per_gc * offset_gc),
#              rlon=slice(265 * pixel_per_gc - pixel_per_gc * offset_gc,
#                         315 * pixel_per_gc + 1 + pixel_per_gc * offset_gc))
# # 30 x 50
# -----------------------------------------------------------------------------
# -> use sub-domain for now...
ds = ds.isel(rlat=slice(150 * pixel_per_gc - pixel_per_gc * offset_gc,
                        540 * pixel_per_gc + 1 + pixel_per_gc * offset_gc),
             rlon=slice(150 * pixel_per_gc - pixel_per_gc * offset_gc,
                        740 * pixel_per_gc + 1 + pixel_per_gc * offset_gc))
# 390 x 590
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
x_ecef, y_ecef, z_ecef = swsg.transform.lonlat2ecef(lon, lat, elevation,
                                                    ellps=ellps)
dem_dim_0, dem_dim_1 = elevation.shape
trans_ecef2enu = swsg.transform.TransformerEcef2enu(
    lon_or=lon.mean(), lat_or=lat.mean(), ellps=ellps)
x_enu, y_enu, z_enu = swsg.transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                              trans_ecef2enu)
del x_ecef, y_ecef, z_ecef

# Test plot
if plot:
    plt.figure()
    plt.pcolormesh(x_enu, y_enu, z_enu)
    plt.colorbar()

# Merge vertex coordinates and pad geometry buffer
vert_grid = swsg.auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)
print("Size of elevation data: %.3f" % (vert_grid.nbytes / (10 ** 9))
      + " GB")
del x_enu, y_enu, z_enu

# Transform 0.0 m surface data (geographic/geodetic -> ENU coordinates)
slice_in = (slice(pixel_per_gc * offset_gc, -pixel_per_gc * offset_gc),
            slice(pixel_per_gc * offset_gc, -pixel_per_gc * offset_gc))
elevation_zero = np.zeros_like(elevation)
x_ecef, y_ecef, z_ecef \
    = swsg.transform.lonlat2ecef(lon[slice_in], lat[slice_in],
                                 elevation_zero[slice_in], ellps=ellps)
dem_dim_in_0, dem_dim_in_1 = elevation_zero[slice_in].shape
x_enu, y_enu, z_enu = swsg.transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                              trans_ecef2enu)
del x_ecef, y_ecef, z_ecef
del lon, lat, elevation

# Test plot
if plot:
    plt.figure()
    plt.pcolormesh(x_enu, y_enu, z_enu)
    plt.colorbar()

# Merge vertex coordinates and pad geometry buffer
vert_grid_in = swsg.auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)
print("Size of elevation data (0.0 m surface): %.3f"
      % (vert_grid_in.nbytes / (10 ** 9)) + " GB")
del x_enu, y_enu, z_enu

# -----------------------------------------------------------------------------
# Compute spatially aggregated correction factors
# -----------------------------------------------------------------------------

# Initialise terrain
terrain = swsg.sun_position.Terrain()
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
    = swsg.transform.lonlat2ecef(subsol_lon, subsol_lat,
                                 subsol_dist, ellps=ellps)
x_enu, y_enu, z_enu = swsg.transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                              trans_ecef2enu)
sun_pos = np.array([x_enu[0], y_enu[0], z_enu[0]], dtype=np.float32)
terrain.sw_dir_cor(sun_pos, sw_dir_cor)
sw_dir_cor_def = sw_dir_cor.copy()  # default
terrain.sw_dir_cor_coherent_rays(sun_pos, sw_dir_cor)
sw_dir_cor_coh = sw_dir_cor.copy()  # coherent rays
print(np.abs(sw_dir_cor_coh - sw_dir_cor_def).max())

# Check output
print("Range of 'sw_dir_cor'-values: [%.2f" % sw_dir_cor_coh.min()
      + ", %.2f" % sw_dir_cor_coh.max() + "]")

# Test plot
levels = np.arange(0.0, 2.0, 0.2)
cmap = cm.roma_r
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False,
                               extend="max")
if plot:
    plt.figure()
    plt.pcolormesh(sw_dir_cor, cmap=cmap, norm=norm)
    plt.colorbar()

# # Compare result with computed lookup table
# ds = xr.open_dataset(dir_work + "SW_dir_cor_lookup.nc")
# ind_lat = np.where(ds["subsolar_lat"].values == subsol_lat)[0][0]
# ind_lon = np.where(ds["subsolar_lon"].values == subsol_lon)[0][0]
# sw_dir_cor_lut = ds["f_cor"][:, :, ind_lat, ind_lon].values
# ds.close()
# dev_abs_max = np.abs(sw_dir_cor_coh - sw_dir_cor_lut).max()
# print("Maximal absolute deviation: %.8f" % dev_abs_max)
# dev_abs_mean = np.abs(sw_dir_cor_coh - sw_dir_cor_lut).mean()
# print("Mean absolute deviation: %.8f" % dev_abs_mean)
