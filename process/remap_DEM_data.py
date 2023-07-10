# Description: Remap MERIT DEM to rotated longitude/latitude grid of model
#              with a target resolution of 3 arc seconds (~90 m). Assume a
#              spherical shape of the Earth. Split large target domains in
#              sub-domains.
#
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
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyinterp
from netCDF4 import Dataset
from packaging import version
import platform
from utilities.grid import grid_frame

mpl.style.use("classic")

# -----------------------------------------------------------------------------
# Specifications for rotated latitude/longitude COSMO domain
# -----------------------------------------------------------------------------

# Alps (with Pyrenees), ~2.2 km
pollat = 43.0          # Latitude of the rotated North Pole [degree]
pollon = -170.0        # Longitude of the rotated North Pole [degree]
polgam = 0.0           # Longitude rotation about the new pole [degree]
ie_tot = 1000          # Number of grid cells (zonal)
je_tot = 600           # Number of grid cells (meridional)
drlon = 0.02           # Grid spacing (zonal) [degree]
drlat = 0.02           # Grid spacing (meridional) [degree]
startrlon_tot = -10.2  # Centre longitude of lower left grid cell [degree]
startrlat_tot = -6.6   # Centre latitude of lower left grid cell [degree]

# # Europe, ~12 km
# pollat = 43.0
# pollon = -170.0
# polgam = 0.0
# ie_tot = 361
# je_tot = 361
# drlon = 0.11
# drlat = 0.11
# startrlon_tot = -23.33
# startrlat_tot = -19.36

# # Switzerland, ~550 m
# pollat = 43.0
# pollon = -170.0
# polgam = 0.0
# ie_tot = 46 * 16
# je_tot = 36 * 16
# drlon = 0.005
# drlat = 0.005
# startrlon_tot = -4.0075 + (0.005 * 12 * 16)
# startrlat_tot = -2.5275 + (0.005 * 12 * 16)

# # Hengduan Mountains, ~550 m
# pollat = 61.81         # Latitude of the rotated North Pole [degree]
# pollon = -81.13        # Longitude of the rotated North Pole [degree]
# polgam = 0.0           # Longitude rotation about the new pole [degre
# ie_tot = 18 * 16       # Number of grid cells (zonal)
# je_tot = 16 * 16       # Number of grid cells (meridional)
# drlon = 0.005          # Grid spacing (zonal) [degree]
# drlat = 0.005          # Grid spacing (meridional) [degree]
# startrlon_tot =-(0.08 * 18 / 2) + (0.005 / 2)
# startrlat_tot =-(0.08 * 16 / 2) + (0.005 / 2)

# # Karakoram, ~550 m
# pollon = -103.49
# pollat = 54.12
# polgam = 0.0
# ie_tot = 18 * 16
# je_tot = 16 * 16
# drlon = 0.005
# drlat = 0.005
# startrlon_tot =-(0.08 * 18 / 2) + (0.005 / 2)
# startrlat_tot =-(0.08 * 16 / 2) + (0.005 / 2)

# # New Zealand, ~550 m
# pollon = 168.10
# pollat = 45.25
# polgam = 180.00
# ie_tot = 14 * 16
# je_tot = 12 * 16
# drlon = 0.005
# drlat = 0.005
# startrlon_tot =-(0.08 * 14 / 2) + (0.005 / 2)
# startrlat_tot =-(0.08 * 12 / 2) + (0.005 / 2)

# -----------------------------------------------------------------------------
# Other settings
# -----------------------------------------------------------------------------

# System-dependent settings
if platform.system() == "Darwin":  # local
    path_dem = "/Users/csteger/Dropbox/IAC/Data/DEMs/MERIT/Tiles/"
    path_work = "/Users/csteger/Desktop/dir_work/"  # working directory
    uzip_dem = True
elif platform.system() == "Linux":  # CSCS
    path_dem = "/store/c2sm/extpar_raw_data/topo/merit/"
    path_work = "/scratch/snx3000/csteger/Subgrid_radiation_data/"
    uzip_dem = False

# Miscellaneous settings
bound_width = 100.0  # width for additional terrain at the boundary [km]
plot_map = True
file_out = "MERIT_remapped_COSMO_%.3f" % drlon + "deg.nc"
extent_max_half = 800.0  # np.inf # maximal 'half' extent of domain [km]
# -> limit maximal absolute value of subsequently computed global ENU
#    coordinates. Should be set to a suitable value regarding 'ray_org_elev'
#    and the representation of ENU coordinates with 32-bit floats.

# Constants
radius_earth = 6_371_229.0  # radius of Earth (according to COSMO/ICON) [m]
dem_res = 3.0 / 3600.0  # resolution of MERIT DEM (3 arc second) [degree]

# -----------------------------------------------------------------------------
# Function(s)
# -----------------------------------------------------------------------------


# Compute required geographic domain
def geo_domain_extent(rlon, rlat, ccrs_rot_pole, ccrs_geo, add_saf=0.1):
    """Computes extent of domain specified in rotated coordinates in geographic
    coordinates.

    Parameters
    ----------
    rlon : ndarray of double
        Array (one-dimensional) with rotated longitude [degree]
    rlat : ndarray of double
        Array (one-dimensional)) with rotated latitudes [degree]
    ccrs_rot_pole : cartopy.crs.RotatedPole
        Cartopy object with rotated pole specifications
    ccrs_geo : cartopy.crs.PlateCarree
        Cartopy object with geographic coordinate system specifications
    add_saf : float
        'Safety' margin [degree]

    Returns
    -------
    geo_extent : tuple of floats
        Extent of geographic domain (lon_min, lon_max, lat_min, lat_max)
        [degree]"""

    rlon_frame, rlat_frame = grid_frame(rlon, rlat, offset=0)
    coord_geo = ccrs_geo.transform_points(ccrs_rot_pole,
                                          rlon_frame, rlat_frame)
    geo_extent = (coord_geo[:, 0].min() - add_saf,
                  coord_geo[:, 0].max() + add_saf,
                  coord_geo[:, 1].min() - add_saf,
                  coord_geo[:, 1].max() + add_saf)
    return geo_extent


# -----------------------------------------------------------------------------
# Determine required DEM tiles and unzip them
# -----------------------------------------------------------------------------

# Ensure that integer number of DEM pixel fit inside a grid cell
if (not (drlon / dem_res).is_integer()) \
        or (not (drlat / dem_res).is_integer()):
    raise ValueError("Model grid spacing is not evenly divisible "
                     + "by DEM grid spacing")

# Compute extended model grid
bound_add = 360.0 / (2.0 * np.pi * radius_earth) * (bound_width * 1000.0)
gc_add_rlat = int(np.ceil(bound_add / drlat))
rlat_model = np.linspace(startrlat_tot - (gc_add_rlat * drlat),
                         startrlat_tot - (gc_add_rlat * drlat)
                         + drlat * (je_tot + 2 * gc_add_rlat - 1),
                         je_tot + 2 * gc_add_rlat, dtype=np.float64)
gc_add_rlon = int(np.ceil(bound_add / drlon))
rlon_model = np.linspace(startrlon_tot - (gc_add_rlon * drlon),
                         startrlon_tot - (gc_add_rlon * drlon)
                         + drlon * (ie_tot + 2 * gc_add_rlon - 1),
                         ie_tot + 2 * gc_add_rlon, dtype=np.float64)
print("Number of grid cells added on each side of domain (rlat/rlon):")
print(str(gc_add_rlat) + ", " + str(gc_add_rlon))

# Compute DEM grid (-> edge coordinates)
pixel_per_gc_x = int(drlon / dem_res)
rlon_edge_dem = np.linspace(rlon_model[0] - drlon / 2.0,
                            rlon_model[-1] + drlon / 2.0,
                            rlon_model.size * pixel_per_gc_x + 1)
pixel_per_gc_y = int(drlat / dem_res)
rlat_edge_dem = np.linspace(rlat_model[0] - drlat / 2.0,
                            rlat_model[-1] + drlat / 2.0,
                            rlat_model.size * pixel_per_gc_y + 1)
print("Maximal absolute deviation in grid spacing:")
print("{:.2e}".format(np.abs(np.diff(rlon_edge_dem) - dem_res).max()))
print("{:.2e}".format(np.abs(np.diff(rlat_edge_dem) - dem_res).max()))
print("Size of interpolated DEM grid: "
      + str(rlon_edge_dem.size) + " x " + str(rlat_edge_dem.size))

# Coordinate systems
globe = ccrs.Globe(datum="WGS84", ellipse="sphere")
ccrs_rot_pole = ccrs.RotatedPole(pole_latitude=pollat, pole_longitude=pollon,
                                 central_rotated_longitude=polgam,
                                 globe=globe)
ccrs_geo = ccrs.PlateCarree(globe=globe)

# Compute required geographical domain
geo_extent = geo_domain_extent(rlon_edge_dem, rlat_edge_dem,
                               ccrs_rot_pole, ccrs_geo)
print("Required geographical extent [degree]:")
print("Longitude: %.2f" % geo_extent[0] + " - %.2f" % geo_extent[1])
print("Latitude: %.2f" % geo_extent[2] + " - %.2f" % geo_extent[3])

# Overview plot
if plot_map:
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs_geo)
    ax.coastlines(zorder=1)
    plt.hlines(y=range(-90, 120, 30), xmin=-180, xmax=180, color="red",
               zorder=2)
    plt.vlines(x=range(-180, 210, 30), ymin=-90, ymax=90, color="red",
               zorder=2)
    plt.axis([geo_extent[0] - 3.0, geo_extent[1] + 3.0,
              geo_extent[2] - 3.0, geo_extent[3] + 3.0])
    rlon_plt, rlat_plt = grid_frame(rlon_edge_dem[0:None:pixel_per_gc_x],
                                    rlat_edge_dem[0:None:pixel_per_gc_y],
                                    offset=0)
    poly = plt.Polygon(list(zip(rlon_plt, rlat_plt)), facecolor="none",
                       edgecolor="blue", transform=ccrs_rot_pole, zorder=3,
                       linewidth=2.0)
    ax.add_patch(poly)
    gl = ax.gridlines(draw_labels=True, linestyle="--", lw=0.5, color="black",
                      zorder=0)
    gl.top_labels = False
    gl.right_labels = False

# Determine required DEM tiles
lon_min = int(np.floor(geo_extent[0] / 30.0) * 30.0)
lon_max = int(np.ceil(geo_extent[1] / 30.0) * 30.0)
lat_min = int(np.floor(geo_extent[2] / 30.0) * 30.0)
lat_max = int(np.ceil(geo_extent[3] / 30.0) * 30.0)
tiles_dem = []
for i in range(lat_min, lat_max, 30):
    for j in range(lon_min, lon_max, 30):
        p1 = "N" + str(i + 30).zfill(2) if (i + 30) >= 0 \
            else "S" + str(abs(i + 30)).zfill(2)
        p2 = "N" + str(i).zfill(2) if i >= 0 else "S" + str(abs(i)).zfill(2)
        p3 = "E" + str(j).zfill(3) if j >= 0 else "W" + str(abs(j)).zfill(3)
        p4 = "E" + str(j + 30).zfill(3) if (j + 30) >= 0 \
            else "W" + str(abs(j + 30)).zfill(3)
        tiles_dem.append("MERIT_" + p1 + "-" + p2 + "_" + p3 + "-" + p4)
        # -> tile name without file ending
print("The following DEM tiles are required:\n" + "\n".join(tiles_dem))

# Unzip DEM tiles (optional)
if uzip_dem:
    cmd = "gunzip -c"
    for i in tiles_dem:
        sf = path_dem + i + ".nc.xz"
        tf = path_work + i + ".nc"
        if not os.path.isfile(tf):
            t_beg = time.time()
            subprocess.call(cmd + " " + sf + " > " + tf, shell=True)
            print("File " + i + ".nc unzipped (%.1f" % (time.time() - t_beg)
                  + " s)")
    path_dem = path_work

# -----------------------------------------------------------------------------
# Regrid DEM bilinearly to model sub-grid
# -----------------------------------------------------------------------------

# Split domain in sub-domains (if required)
if np.isinf(extent_max_half):
    len_max_y = 32_767
    len_max_x = 32_767
else:
    extent_max = 360.0 / (2.0 * np.pi * radius_earth) \
                 * (extent_max_half * 1000.0 * 2.0)
    len_max_y = np.minimum(int(np.ceil(extent_max / (drlat / pixel_per_gc_y))),
                         32_767)
    len_max_x = np.minimum(int(np.ceil(extent_max / (drlon / pixel_per_gc_x))),
                         32_767)
# 32_767: maximal length along one dimension determined by Embree restriction
num_subdom_y = int(np.ceil(rlat_edge_dem.size / len_max_y))
num_subdom_x = int(np.ceil(rlon_edge_dem.size / len_max_x))
flag_subdom = (num_subdom_y > 1) or (num_subdom_x > 1)
if flag_subdom:
    print("DEM domain is split in sub-domains: ("
          + str(num_subdom_y) + ", " + str(num_subdom_x) + ")")

# Loop through subdomains
ind_y = np.linspace(gc_add_rlat, gc_add_rlat + je_tot,
                    (num_subdom_y + 1), dtype=int) * pixel_per_gc_y
ind_x = np.linspace(gc_add_rlon,  gc_add_rlon + ie_tot,
                    (num_subdom_x + 1), dtype=int) * pixel_per_gc_x
num_pixel_y = pixel_per_gc_y * gc_add_rlat
num_pixel_x = pixel_per_gc_x * gc_add_rlon
for i in range(num_subdom_y):
    for j in range(num_subdom_x):

        # Compute required geographical domain
        slice_y = slice(ind_y[i] - num_pixel_y, ind_y[i + 1] + num_pixel_y + 1)
        slice_x = slice(ind_x[j] - num_pixel_x, ind_x[j + 1] + num_pixel_x + 1)
        rlat_edge_dem_sd = rlat_edge_dem[slice_y]
        rlon_edge_dem_sd = rlon_edge_dem[slice_x]
        geo_extent = geo_domain_extent(rlon_edge_dem_sd,
                                       rlat_edge_dem_sd,
                                       ccrs_rot_pole, ccrs_geo)
        rlat_model_sd = rlat_model[int(slice_y.start / pixel_per_gc_y)
                                   :int((slice_y.stop - 1) / pixel_per_gc_y)]
        rlon_model_sd = rlon_model[int(slice_x.start / pixel_per_gc_x)
                                   :int((slice_x.stop - 1) / pixel_per_gc_x)]

        # Merge DEM tiles and crop domain
        ds = xr.open_mfdataset([path_dem + i + ".nc" for i in tiles_dem])
        ds = ds.sel(lon=slice(geo_extent[0], geo_extent[1]),
                    lat=slice(geo_extent[3], geo_extent[2]))
        lon = ds["lon"].values.astype(np.float64)  # geographic longitude [deg]
        lat = ds["lat"].values.astype(np.float64)  # geographic latitude [deg]
        elevation = ds["Elevation"].values  # np.float32
        print(elevation.shape)
        ds.close()

        # Set elevation of sea grid cells to 0.0 m
        mask_water = np.ones_like(elevation)
        mask_water[~np.isnan(elevation)] = np.nan
        elevation[np.isnan(elevation)] = 0.0

        # Bilinear interpolation in blocks
        t_beg = time.time()
        x_axis = pyinterp.Axis(lon)
        y_axis = pyinterp.Axis(lat)
        grid_elev = pyinterp.Grid2D(y_axis, x_axis, elevation)
        grid_lsm = pyinterp.Grid2D(y_axis, x_axis, mask_water)
        block_size = 5000
        elevation_ip = np.empty((rlat_edge_dem_sd.size,
                                 rlon_edge_dem_sd.size),
                                dtype=np.float32)
        mask_water_ip = np.empty((rlat_edge_dem_sd.size,
                                  rlon_edge_dem_sd.size),
                                 dtype=np.float32)
        lon_ip = np.empty_like(elevation_ip)
        lat_ip = np.empty_like(elevation_ip)
        steps_rlat = int(np.ceil(rlat_edge_dem_sd.size / block_size))
        steps_rlon = int(np.ceil(rlon_edge_dem_sd.size / block_size))
        print("Number of steps in block-wise interpolation: "
              + str(steps_rlat * steps_rlon))
        for k in range(steps_rlat):
            for m in range(steps_rlon):
                slic = (slice(k * block_size, (k + 1) * block_size),
                        slice(m * block_size, (m + 1) * block_size))
                rlon, rlat = np.meshgrid(rlon_edge_dem_sd[slic[1]],
                                         rlat_edge_dem_sd[slic[0]])
                coord = ccrs_geo.transform_points(ccrs_rot_pole, rlon, rlat)
                y_ip = coord[:, :, 1].ravel()
                x_ip = coord[:, :, 0].ravel()
                elevation_ip[slic] \
                    = pyinterp.bivariate(grid_elev, y_ip, x_ip,
                                         interpolator="bilinear",
                                         bounds_error=True, num_threads=0) \
                    .reshape(rlon.shape)
                mask_water_ip[slic] \
                    = pyinterp.bivariate(grid_lsm, y_ip, x_ip,
                                         interpolator="bilinear",
                                         bounds_error=True, num_threads=0) \
                    .reshape(rlon.shape)
                lon_ip[slic] = coord[:, :, 0]
                lat_ip[slic] = coord[:, :, 1]
                print("Step " + str((k * steps_rlon) + m + 1)
                      + " of " + str(steps_rlat * steps_rlon) + " completed")
        print("Bilinear interpolation completed (%.1f" % (time.time() - t_beg)
              + " s)")

        # Save to NetCDF file
        if not flag_subdom:
            file_out_p = file_out
        else:
            file_out_p = file_out[:-3] + "_y" + str(i) + "_x" + str(j) + ".nc"
        ncfile = Dataset(filename=path_work + file_out_p,  mode="w")
        ncfile.num_grid_cells_inner_zonal = ie_tot
        ncfile.num_grid_cells_inner_meridional = je_tot
        ncfile.pixels_per_grid_cell_zonal = pixel_per_gc_x
        ncfile.pixels_per_grid_cell_meridional = pixel_per_gc_y
        ncfile.offset_grid_cells_zonal = gc_add_rlon
        ncfile.offset_grid_cells_meridional = gc_add_rlat
        ncfile.createDimension(dimname="rlat", size=elevation_ip.shape[0])
        ncfile.createDimension(dimname="rlon", size=elevation_ip.shape[1])
        # ---------------------------------------------------------------------
        nc_rlat = ncfile.createVariable(varname="rlat", datatype="f",
                                        dimensions="rlat")
        nc_rlat[:] = rlat_edge_dem_sd
        nc_rlat.long_name = "latitude in rotated pole grid"
        nc_rlat.units = "degrees"
        nc_rlon = ncfile.createVariable(varname="rlon", datatype="f",
                                        dimensions="rlon")
        nc_rlon[:] = rlon_edge_dem_sd
        nc_rlon.long_name = "longitude in rotated pole grid"
        nc_rlon.units = "degrees"
        # ---------------------------------------------------------------------
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
        # ---------------------------------------------------------------------
        nc_data = ncfile.createVariable(varname="Elevation", datatype="f",
                                        dimensions=("rlat", "rlon"))
        nc_data[:] = elevation_ip
        nc_data.units = "m"
        # ---------------------------------------------------------------------
        nc_data = ncfile.createVariable(varname="mask_water", datatype="i1",
                                        dimensions=("rlat", "rlon"))
        nc_data[:] = np.isfinite(mask_water_ip).astype(np.int8)
        nc_data.units = "m"
        # ---------------------------------------------------------------------
        nc_meta = ncfile.createVariable("rotated_pole", "S1",)
        nc_meta.grid_mapping_name = "rotated_latitude_longitude"
        nc_meta.grid_north_pole_longitude = pollon
        nc_meta.grid_north_pole_latitude = pollat
        nc_meta.north_pole_grid_longitude = polgam
        # ---------------------------------------------------------------------
        ncfile.createDimension(dimname="rlat_gc", size=rlat_model_sd.size)
        nc_rlat = ncfile.createVariable(varname="rlat_gc", datatype="f",
                                        dimensions="rlat_gc")
        nc_rlat[:] = rlat_model_sd
        nc_rlat.long_name = "latitude of grid cells in rotated pole grid"
        nc_rlat.units = "degrees"
        ncfile.createDimension(dimname="rlon_gc", size=rlon_model_sd.size)
        nc_rlon = ncfile.createVariable(varname="rlon_gc", datatype="f",
                                        dimensions="rlon_gc")
        nc_rlon[:] = rlon_model_sd
        nc_rlon.long_name = "longitude of grid cells in rotated pole grid"
        nc_rlon.units = "degrees"
        # ---------------------------------------------------------------------
        ncfile.close()
