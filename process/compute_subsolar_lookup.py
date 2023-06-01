# Description: Compute subgrid correction factors for direct downward shortwave
#              radiation (aggregated to the model grid cell resolution) for an
#              array of sun positions
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

# Grid for subsolar points
# subsol_lon = np.linspace(-180.0, 174.0,60, dtype=np.float64)  # 6 degree
# subsol_lat = np.linspace(-23.5, 23.5, 21, dtype=np.float64)  # 2.35 degree
subsol_lon = np.linspace(-180.0, 172.0, 45, dtype=np.float64)  # 8 degree
subsol_lat = np.linspace(-23.5, 23.5, 15, dtype=np.float64)  # 3.36 degree

# Ray-tracing and 'SW_dir_cor' calculation
dist_search = 100.0  # search distance for terrain shading [kilometre]
geom_type = "quad"  # "quad" ca. 25% faster than "grid"
ang_max = 89.5
sw_dir_cor_max = 20.0

# Miscellaneous settings
dir_work = "/Users/csteger/Desktop/dir_work/"  # working directory
ellps = "sphere"  # Earth's surface approximation (sphere, GRS80 or WGS84)
plot = True

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
# # sub-domain with reduces boundary: 30 x 50 (complex terrain)
# offset_gc = int(offset_gc / 2)
# ds = ds.isel(rlat=slice(325 * pixel_per_gc - pixel_per_gc * offset_gc,
#                         355 * pixel_per_gc + 1 + pixel_per_gc * offset_gc),
#              rlon=slice(265 * pixel_per_gc - pixel_per_gc * offset_gc,
#                         315 * pixel_per_gc + 1 + pixel_per_gc * offset_gc))
# -----------------------------------------------------------------------------
# sub-domain with reduces boundary: 30 x 50 (~flat terrain)
offset_gc = int(offset_gc / 2)
ds = ds.isel(rlat=slice(375 * pixel_per_gc - pixel_per_gc * offset_gc,
                        405 * pixel_per_gc + 1 + pixel_per_gc * offset_gc),
             rlon=slice(65 * pixel_per_gc - pixel_per_gc * offset_gc,
                        115 * pixel_per_gc + 1 + pixel_per_gc * offset_gc))
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

# Transform locations of subsolar points
subsol_lon_2d, subsol_lat_2d = np.meshgrid(subsol_lon, subsol_lat)
subsol_dist_2d = np.empty(subsol_lon_2d.shape, dtype=np.float32)
subsol_dist_2d[:] = Distance(au=1).m
# astronomical unit (~average Sun-Earth distance) [m]
x_ecef, y_ecef, z_ecef \
    = swsg.transform.lonlat2ecef(subsol_lon_2d, subsol_lat_2d,
                                 subsol_dist_2d, ellps=ellps)
x_enu, y_enu, z_enu = swsg.transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                              trans_ecef2enu)

# Test plot
if plot:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(0.0, 0.0, 0.0 , marker=".", s=100, color="lightblue")
    # ax.scatter(x_ecef.ravel(), y_ecef.ravel(), z_ecef.ravel() , marker=".",
    #            s=20, color="orangered")
    # ax.set_xlabel("x-axis")
    # ax.set_ylabel("y-axis")
    # ax.set_zlabel("z-axis")
    ax.scatter(x_enu.ravel(), y_enu.ravel(), z_enu.ravel() , marker=".",
               s=20, color="orangered")
    ax.set_xlabel("East")
    ax.set_ylabel("North")
    ax.set_zlabel("Up")

# Combine sun position in one array
sun_pos = np.concatenate((x_enu[:, :, np.newaxis],
                          y_enu[:, :, np.newaxis],
                          z_enu[:, :, np.newaxis]), axis=2)

# -----------------------------------------------------------------------------
# Compute spatially aggregated correction factors
# -----------------------------------------------------------------------------

# Compute
sw_dir_cor_def = swsg.subsolar_lookup.sw_dir_cor(
    vert_grid, dem_dim_0, dem_dim_1,
    vert_grid_in, dem_dim_in_0, dem_dim_in_1,
    sun_pos, pixel_per_gc, offset_gc,
    dist_search=dist_search, geom_type=geom_type,
    ang_max=ang_max, sw_dir_cor_max=sw_dir_cor_max)
sw_dir_cor_coh = swsg.subsolar_lookup.sw_dir_cor_coherent_rays(
    vert_grid, dem_dim_0, dem_dim_1,
    vert_grid_in, dem_dim_in_0, dem_dim_in_1,
    sun_pos, pixel_per_gc, offset_gc,
    dist_search=dist_search, geom_type=geom_type,
    ang_max=ang_max, sw_dir_cor_max=sw_dir_cor_max)
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
    ind_2, ind_3 = 0, 22
    plt.figure(figsize=(13, 7))
    plt.pcolormesh(sw_dir_cor_coh[:, :, ind_2, ind_3], cmap=cmap, norm=norm)
    plt.colorbar()

# Test Plot
if plot:
    ind_0, ind_1 = 7, 17
    plt.figure(figsize=(14, 6))
    plt.pcolormesh(subsol_lon, subsol_lat, sw_dir_cor_coh[ind_0, ind_1, :, :],
                   cmap=cmap, norm=norm)
    plt.xticks(range(-180, 200, 20), range(-180, 200, 20))
    plt.yticks(range(-25, 30, 5), range(-25, 30, 5))
    plt.axis([subsol_lon[0], subsol_lon[-1], subsol_lat[0], subsol_lat[-1]])
    plt.xlabel("Subsolar longitude [degree]")
    plt.ylabel("Subsolar latitude [degree]")
    plt.colorbar()

# Select relevant subsolar longitude range (add -/+ 1)
mask = (sw_dir_cor_coh.sum(axis=(0, 1, 2)) != 0)
slic = slice(np.maximum(np.where(mask)[0][0] - 1, 0),
             np.minimum(np.where(mask)[0][-1] + 2, len(mask)))
print(mask[slic])
print("Size of lookup table: %.2f"
      % (sw_dir_cor_coh[:, :, :, slic].nbytes / (10 ** 6)) + " MB")

# Save to NetCDF file
ncfile = Dataset(filename=dir_work + "SW_dir_cor_lookup.nc", mode="w")
ncfile.ang_max = "%.2f" % ang_max + " degrees"
ncfile.sw_dir_cor_max = "%.2f" % sw_dir_cor_max
# -----------------------------------------------------------------------------
ncfile.createDimension(dimname="gc_lat", size=sw_dir_cor_coh.shape[0])
ncfile.createDimension(dimname="gc_lon", size=sw_dir_cor_coh.shape[1])
ncfile.createDimension(dimname="subsolar_lat", size=sw_dir_cor_coh.shape[2])
ncfile.createDimension(dimname="subsolar_lon", size=mask[slic].size)
# -----------------------------------------------------------------------------
nc_sslat = ncfile.createVariable(varname="subsolar_lat", datatype="f",
                                dimensions="subsolar_lat")
nc_sslat[:] = subsol_lat
nc_sslat.long_name = "subsolar latitude"
nc_sslat.units = "degrees"
nc_sslon = ncfile.createVariable(varname="subsolar_lon", datatype="f",
                                dimensions="subsolar_lon")
nc_sslon[:] = subsol_lon[slic]
nc_sslon.long_name = "subsolar longitude"
nc_sslon.units = "degrees"
# -----------------------------------------------------------------------------
nc_data = ncfile.createVariable(varname="f_cor", datatype="f",
                                dimensions=("gc_lat", "gc_lon",
                                            "subsolar_lat", "subsolar_lon"))
nc_data[:] = sw_dir_cor_coh[:, :, :, slic]
nc_data.units = "-"
# -----------------------------------------------------------------------------
ncfile.close()

# Test load
ds = xr.open_dataset(dir_work + "SW_dir_cor_lookup.nc")
f_cor = ds["f_cor"].values
ds.close()
