# Description: Remap MERIT DEM to rotated longitude/latitude grid of regional
#              climate model with a target resolution of 3 arc seconds (~90 m).
#              Assume a spherical shape of the Earth.
# Notes:
# - limitation: script only works correctly if required interpolation domain
#   does not cross the true North/South Pole or the -/+180 degree meridian
# - MERIT DEM data can be downloaded from:
#   http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/
#
# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
import xarray as xr
import subprocess
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
from scipy.interpolate import RegularGridInterpolator
from netCDF4 import Dataset
from packaging import version
from utilities.grid import grid_frame

mpl.style.use("classic")

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Regional climate model grid specification (-> rotated lon/lat COSMO grid)
pollat = 43.0 			# Latitude of the rotated North Pole
pollon = -170.0 		# Longitude of the rotated North Pole
ie_tot = 800			# Number of grid cells (zonal)
je_tot = 600			# Number of grid cells (meridional)
dlon = 0.02				# Grid spacing (zonal)
dlat = 0.02				# Grid spacing (meridional)
startlon_tot = -6.2     # Centre longitude of lower left grid cell
startlat_tot = -6.6     # Centre latitude of lower left grid cell

# Miscellaneous settings
bound_width = 100.0     # Width for additional terrain at the boundary [km]
path_merit = "/Users/csteger/Dropbox/IAC/Data/DEMs/MERIT/Tiles/"
# path_merit = "/store/c2sm/extpar_raw_data/topo/merit/"  # CSCS
dir_work = "/Users/csteger/Desktop/dir_work/"  # working directory

# Constants
radius_earth = 6371229.0  # radius of Earth (according to COSMO/ICON) [m]
merit_res = 3.0 / 3600.0  # resolution of MERIT DEM (3 arc second) [degree]

# -----------------------------------------------------------------------------
# Determine required MERIT tiles
# -----------------------------------------------------------------------------

# Check SciPy version
if version.parse(scipy.__version__) < version.parse("1.10.0"):
    raise ImportError("SciPy version 1.10.0 required. Excessive memory "
                      + "requirements of function 'RegularGridInterpolator'"
                        "in older versions")

# Compute extended COSMO grid
bound_add = 360.0 / (2.0 * np.pi * radius_earth) * (bound_width * 1000.0)
gc_add_lon = int(np.ceil(bound_add / dlon))
rlon_cosmo = np.linspace(startlon_tot - (gc_add_lon * dlon),
                         startlon_tot - (gc_add_lon * dlon)
                         + dlon * (ie_tot + 2 * gc_add_lon - 1),
                         ie_tot + 2 * gc_add_lon, dtype=np.float64)
gc_add_lat = int(np.ceil(bound_add / dlat))
rlat_cosmo = np.linspace(startlat_tot - (gc_add_lat * dlat),
                         startlat_tot - (gc_add_lat * dlat)
                         + dlat * (je_tot + 2 * gc_add_lat - 1),
                         je_tot + 2 * gc_add_lat, dtype=np.float64)

# Compute MERIT grid (-> edge coordinates)
if (not (dlon / merit_res).is_integer()) \
        or (not (dlat / merit_res).is_integer()):
    raise ValueError("COSMO grid spacing is not evenly divisible "
                     + "by MERIT grid spacing")
rlon_edge_merit = np.linspace(rlon_cosmo[0] - dlon / 2.0,
                              rlon_cosmo[-1] + dlon / 2.0,
                              rlon_cosmo.size * int(dlon / merit_res) + 1)
rlat_edge_merit = np.linspace(rlat_cosmo[0] - dlat / 2.0,
                              rlat_cosmo[-1] + dlat / 2.0,
                              rlat_cosmo.size * int(dlat / merit_res) + 1)
print("Maximal absolute deviation in grid spacing:")
print("{:.2e}".format(np.abs(np.diff(rlon_edge_merit) - merit_res).max()))
print("{:.2e}".format(np.abs(np.diff(rlat_edge_merit) - merit_res).max()))
print("Size of interpolated MERIT grid: "
      + str(rlon_edge_merit.size) + " x " + str(rlat_edge_merit.size))

# Determine required MERIT tiles
rlon_frame, rlat_frame = grid_frame(rlon_edge_merit, rlat_edge_merit, offset=0)
globe = ccrs.Globe(datum="WGS84", ellipse="sphere")
ccrs_rot_pole = ccrs.RotatedPole(pole_latitude=pollat, pole_longitude=pollon,
                                 globe=globe)
ccrs_geo = ccrs.PlateCarree(globe=globe)
coord = ccrs_geo.transform_points(ccrs_rot_pole, rlon_frame, rlat_frame)
add = 0.1  # safety margin [degree]
geo_extent = [coord[:, 0].min() - add, coord[:, 0].max() + add,
              coord[:, 1].min() - add, coord[:, 1].max() + add]
# (lon_min, lon_max, lat_min, lat_max)
print("Required geographical extent [degree]:")
print("Longitude: %.2f" % geo_extent[0] + " - %.2f" % geo_extent[1])
print("Latitude: %.2f" % geo_extent[2] + " - %.2f" % geo_extent[3])
lon_min = int(np.floor(geo_extent[0] / 30.0) * 30.0)
lon_max = int(np.ceil(geo_extent[1] / 30.0) * 30.0)
lat_min = int(np.floor(geo_extent[2] / 30.0) * 30.0)
lat_max = int(np.ceil(geo_extent[3] / 30.0) * 30.0)
tiles_merit = []
for i in range(lat_min, lat_max, 30):
    for j in range(lon_min, lon_max, 30):
        p1 = "N" + str(i + 30).zfill(2) if (i + 30) >= 0 \
            else "S" + str(abs(i + 30)).zfill(2)
        p2 = "N" + str(i).zfill(2) if i >= 0 else "S" + str(abs(i)).zfill(2)
        p3 = "E" + str(j).zfill(3) if j >= 0 else "W" + str(abs(j)).zfill(3)
        p4 = "E" + str(j + 30).zfill(3) if (j + 30) >= 0 \
            else "W" + str(abs(j + 30)).zfill(3)
        tiles_merit.append("MERIT_" + p1 + "-" + p2 + "_" + p3 + "-" + p4)
        # -> tile name without file ending

# -----------------------------------------------------------------------------
# Regrid MERIT DEM bilinearly
# -----------------------------------------------------------------------------

# Unzip MERIT tiles
cmd = "gunzip -c"
for i in tiles_merit:
    sf = path_merit + i + ".nc.xz"
    tf = dir_work + i + ".nc"
    subprocess.call(cmd + " " + sf + " > " + tf, shell=True)
    print("File " + i + ".nc unzipped")

# Merge MERIT tiles and crop domain
time.sleep(1.0)
ds = xr.open_mfdataset([dir_work + i + ".nc" for i in tiles_merit])
ds = ds.sel(lon=slice(geo_extent[0], geo_extent[1]),
            lat=slice(geo_extent[3], geo_extent[2]))
lon = ds["lon"].values.astype(np.float64)  # geographic longitude [degree]
lat = ds["lat"].values.astype(np.float64)   # geographic latitude [degree]
elevation = ds["Elevation"].values.astype(np.float64)  # ~5.0 GB
# -> function 'RegularGridInterpolator' only works with np.float64
ds.close()

# Set elevation of sea grid cells to 0.0 m
elevation[np.isnan(elevation)] = 0.0

# Interpolation (bilinear) in blocks
f_ip = RegularGridInterpolator((lat, lon), elevation, method="linear",
                               bounds_error=True)
block_size = 5000
elevation_ip = np.empty((rlat_edge_merit.size, rlon_edge_merit.size),
                        dtype=np.float32)
lon_ip = np.empty_like(elevation_ip)
lat_ip = np.empty_like(elevation_ip)
steps_rlat = int(np.ceil(rlat_edge_merit.size / block_size))
steps_rlon = int(np.ceil(rlon_edge_merit.size / block_size))
for i in range(steps_rlat):
    for j in range(steps_rlon):
        slic = (slice(i * block_size, (i + 1) * block_size),
                slice(j * block_size, (j + 1) * block_size))
        rlon, rlat = np.meshgrid(rlon_edge_merit[slic[1]],
                                 rlat_edge_merit[slic[0]])
        coord = ccrs_geo.transform_points(ccrs_rot_pole, rlon, rlat)
        pts_ip = np.vstack((coord[:, :, 1].ravel(), coord[:, :, 0].ravel())) \
            .transpose()
        elevation_ip[slic] = f_ip(pts_ip).reshape(rlon.shape)
        lon_ip[slic] = coord[:, :, 0]
        lat_ip[slic] = coord[:, :, 1]
        print("Step " + str((i * steps_rlon) + j + 1)
              + " of " + str(steps_rlat * steps_rlon) + " completed")

# Save to NetCDF file
ncfile = Dataset(filename=dir_work + "MERIT_remapped_COSMO.nc", mode="w")
ncfile.sub_grid_info = "MERIT pixels per COSMO grid cell"
ncfile.sub_grid_info_zonal = int(dlon / merit_res)
ncfile.sub_grid_info_meridional = int(dlat / merit_res)
ncfile.offset_grid_cells_zonal = gc_add_lon
ncfile.offset_grid_cells_meridional = gc_add_lat
ncfile.createDimension(dimname="rlat", size=elevation_ip.shape[0])
ncfile.createDimension(dimname="rlon", size=elevation_ip.shape[1])
# -----------------------------------------------------------------------------
nc_rlat = ncfile.createVariable(varname="rlat", datatype="f",
                                dimensions="rlat")
nc_rlat[:] = rlat_edge_merit
nc_rlat.long_name = "latitude in rotated pole grid"
nc_rlat.units = "degrees"
nc_rlon = ncfile.createVariable(varname="rlon", datatype="f",
                                dimensions="rlon")
nc_rlon[:] = rlon_edge_merit
nc_rlon.long_name = "longitude in rotated pole grid"
nc_rlon.units = "degrees"
# -----------------------------------------------------------------------------
nc_lat = ncfile.createVariable(varname="lat", datatype="f",
                               dimensions=("rlat", "rlon"))
nc_lat[:] = lat_ip
nc_lat.long_name = "geographical latitude"
nc_lat.units = "degrees_north"
nc_lon = ncfile.createVariable(varname="lon", datatype="f",
                               dimensions=("rlat", "rlon"))
nc_lon[:] = lon_ip
nc_lon.long_name = "geographical longitude"
nc_lon.units = "degrees_east"
# -----------------------------------------------------------------------------
nc_data = ncfile.createVariable(varname="Elevation", datatype="f",
                                dimensions=("rlat", "rlon"))
nc_data[:] = elevation_ip
nc_data.units = "m"
# -----------------------------------------------------------------------------
nc_meta = ncfile.createVariable("rotated_pole", "S1",)
nc_meta.grid_mapping_name = "rotated_latitude_longitude"
nc_meta.grid_north_pole_longitude = pollon
nc_meta.grid_north_pole_latitude = pollat
nc_meta.north_pole_grid_longitude = 0.0
# -----------------------------------------------------------------------------
ncfile.close()

time.sleep(1.0)
for i in [dir_work + i + ".nc" for i in tiles_merit]:
    os.remove(i)

# -----------------------------------------------------------------------------
# Test loading of remapped MERIT data
# -----------------------------------------------------------------------------

# Load data for sub-region
ds = xr.open_dataset(dir_work + "MERIT_remapped_COSMO.nc")
# ds = ds.isel(rlon=slice(6600, 7500), rlat=slice(8100, 8700))
ds = ds.isel(rlon=slice(6750, 7050), rlat=slice(8400, 8650))
rlon = ds["rlon"].values
rlat = ds["rlat"].values
lon = ds["lon"].values
lat = ds["lat"].values
elevation = ds["Elevation"].values
pole_lon = ds["rotated_pole"].grid_north_pole_longitude
pole_lat = ds["rotated_pole"].grid_north_pole_latitude
num_gc = ds.attrs["sub_grid_info_zonal"]
offset = ds.attrs["offset_grid_cells_zonal"]
ds.close()

# Reference points (-> check if geo-referencing correct...)
ref_points = {"Eiger": (46.577608, 8.005287),
              "Moench": (46.55849, 7.9973),
              "Jungfrau": (46.538, 7.9637),
              "Schreckhorn": (46.589139, 8.118536),
              "Finsteraarhorn": (46.537222, 8.126111)}

# Test plot
ccrs_rot_pole = ccrs.RotatedPole(pole_latitude=pole_lat,
                                 pole_longitude=pole_lon)
plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
plt.pcolormesh(rlon, rlat, elevation, transform=ccrs_rot_pole)
plt.colorbar()
ax.add_feature(cfeature.BORDERS.with_scale("10m"), linestyle="-")
for i in ref_points.keys():
    plt.scatter(ref_points[i][1], ref_points[i][0], marker="^", s=50,
                color="black", transform=ccrs.PlateCarree())
ax.set_aspect("auto")
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.5, color="black",
                  alpha=0.8, linestyle="-", draw_labels=True)
gl.top_labels = False
gl.right_labels = False

# Load inner domain (without boundary width)
ds = xr.open_dataset(dir_work + "MERIT_remapped_COSMO.nc")
offset = ds.attrs["offset_grid_cells_zonal"] * ds.attrs["sub_grid_info_zonal"]
print("Size of full domain: " + "%.2f" % (ds["Elevation"].nbytes
                                          / (10 ** 9) * 3) + " GB")
ds = ds.isel(rlon=slice(offset, -offset), rlat=slice(offset, -offset))
print("Size of inner domain: " + "%.2f" % (ds["Elevation"].nbytes
                                           / (10 ** 9) * 3) + " GB")
ds.close()
