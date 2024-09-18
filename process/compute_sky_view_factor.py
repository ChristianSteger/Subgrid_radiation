# Description: Computation of sky view factor and related quantities with slope
#              (aggregated to the model grid cell resolution)
#
# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import time
import numpy as np
import xarray as xr
from skyfield.api import Distance
from netCDF4 import Dataset
import glob
import platform
from subgrid_radiation import transform, auxiliary
from subgrid_radiation.sun_position_array import horizon
from subgrid_radiation import sun_position_array

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Ray-tracing and sky view factor calculation
dist_search = 100.0  # search distance for terrain shading [kilometre]
geom_type = "grid"  # "grid" or "quad"
# hori_azim_num, hori_acc = 30, 3.0
# hori_azim_num, hori_acc = 45, 2.0
# hori_azim_num, hori_acc = 60, 1.5
hori_azim_num, hori_acc = 90, 1.0
# hori_azim_num, hori_acc = 180, 0.5
# hori_azim_num, hori_acc = 360, 0.25
ray_algorithm = "guess_constant"
elev_ang_low_lim = -85.0  # -15.0

# File input/output
# file_in = "MERIT_remapped_COSMO_0.020deg_y?_x?.nc"
# file_in = "MERIT_remapped_COSMO_0.020deg.nc"
# file_in = "MERIT_remapped_COSMO_0.010deg_y?_x?.nc"
# file_in = "MERIT_remapped_COSMO_0.005deg.nc"
file_in = "MERIT_remapped_COSMO_0.005deg_y?_x?.nc"

# Miscellaneous settings
path_work = {"local": "/Users/csteger/Desktop/dir_work/",
             "cscs": "/scratch/snx3000/csteger/Subgrid_radiation/Output/"}
radius_earth = 6_371_229.0  # radius of Earth (according to COSMO/ICON) [m]

# -----------------------------------------------------------------------------
# Process data
# -----------------------------------------------------------------------------

# Set working path depending on system
systems = {"Darwin": "local", "Linux": "cscs"}
path_work = path_work[systems[platform.system()]]

# Loop through subdomains
files_in = glob.glob(path_work + file_in)
files_in.sort()
print("Number of sub-domains: " + str(len(files_in)))
for i in files_in:

    print((" Process file " + i.split("/")[-1] + " ").center(79, "-"))

    # Load small data
    ds = xr.open_dataset(i)
    pixel_per_gc = ds.attrs["pixels_per_grid_cell_zonal"]
    # pixel per grid cell (along one dimension)
    offset_gc = ds.attrs["offset_grid_cells_zonal"]
    # offset in number of grid cells
    pole_lon = ds["rotated_pole"].grid_north_pole_longitude
    pole_lat = ds["rotated_pole"].grid_north_pole_latitude
    rlon_gc = ds["rlon_gc"].values
    rlat_gc = ds["rlat_gc"].values
    ds.close()

    # Compute vertices of DEM triangles in global ENU coordinates
    t_beg = time.perf_counter()
    ds = xr.open_dataset(i)
    lon = ds["lon"].values.astype(np.float64)
    lat = ds["lat"].values.astype(np.float64)
    elevation = ds["Elevation"].values.astype(np.float64)
    ds.close()
    dem_dim_0, dem_dim_1 = elevation.shape
    trans_lonlat2enu = transform.TransformerLonlat2enu(
        lon_or=lon.mean(), lat_or=lat.mean(), radius_earth=radius_earth)
    transform.lonlat2ecef(lon, lat, elevation, trans_lonlat2enu, in_place=True)
    transform.ecef2enu(lon, lat, elevation, trans_lonlat2enu, in_place=True)
    x_enu = lon.astype(np.float32)
    del lon
    y_enu = lat.astype(np.float32)
    del lat
    z_enu = elevation.astype(np.float32)
    del elevation
    vert_grid = auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)
    print("Size of elevation data: %.3f" % (vert_grid.nbytes / (10 ** 9))
          + " GB")
    del x_enu, y_enu, z_enu
    print("DEM vertices data prepared (%.1f" % (time.perf_counter() - t_beg)
          + " s)")

    # Compute vertices of '0.0 m surface' triangles in global ENU coordinates
    t_beg = time.perf_counter()
    slice_in = (slice(pixel_per_gc * offset_gc, -pixel_per_gc * offset_gc),
                slice(pixel_per_gc * offset_gc, -pixel_per_gc * offset_gc))
    ds = xr.open_dataset(i)
    ds = ds.isel(rlat=slice_in[0], rlon=slice_in[1])
    lon = ds["lon"].values.astype(np.float64)
    lat = ds["lat"].values.astype(np.float64)
    ds.close()
    elevation_zero = np.zeros_like(lon)
    dem_dim_in_0, dem_dim_in_1 = elevation_zero.shape
    transform.lonlat2ecef(lon, lat, elevation_zero, trans_lonlat2enu,
                          in_place=True)
    transform.ecef2enu(lon, lat, elevation_zero, trans_lonlat2enu,
                       in_place=True)
    value_abs_max = np.array([np.abs(lon).max(), np.abs(lat).max(),
                              np.abs(elevation_zero).max()]).max()
    print("Maximal absolute ENU coordinate value (32-bit float) "
          + "in inner domain: %.2f" % value_abs_max)
    x_enu = lon.astype(np.float32)
    del lon
    y_enu = lat.astype(np.float32)
    del lat
    z_enu = elevation_zero.astype(np.float32)
    del elevation_zero
    vert_grid_in = auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)
    print("Size of elevation data (0.0 m surface): %.3f"
          % (vert_grid_in.nbytes / (10 ** 9)) + " GB")
    del x_enu, y_enu, z_enu
    print("'0.0 m surface' vertices data prepared (%.1f"
          % (time.perf_counter() - t_beg) + " s)")

    # Mask (optional)
    num_gc_y = int((dem_dim_0 - 1) / pixel_per_gc) - 2 * offset_gc
    num_gc_x = int((dem_dim_1 - 1) / pixel_per_gc) - 2 * offset_gc
    mask = np.zeros((num_gc_y, num_gc_x), dtype=np.uint8)
    # mask[-50:, -50:] = 1
    # mask[:3000, :3000] = 1
    mask[:] = 1

    # Ray-tracing
    sky_view_factor, area_increase_factor, sky_view_area_factor, \
        slope, aspect \
        = sun_position_array.horizon.sky_view_factor(
            vert_grid, dem_dim_0, dem_dim_1,
            vert_grid_in, dem_dim_in_0, dem_dim_in_1,
            trans_lonlat2enu.north_pole_enu, pixel_per_gc, offset_gc,
            mask=mask, dist_search=dist_search, hori_azim_num=hori_azim_num,
            hori_acc=hori_acc, ray_algorithm=ray_algorithm,
            elev_ang_low_lim=elev_ang_low_lim, geom_type=geom_type)

    # _, _, _, _, _, distance \
    #     = sun_position_array.horizon.sky_view_factor_dist(
    #     vert_grid, dem_dim_0, dem_dim_1,
    #     vert_grid_in, dem_dim_in_0, dem_dim_in_1,
    #     trans_lonlat2enu.north_pole_enu, pixel_per_gc, offset_gc,
    #     mask=mask, dist_search=dist_search,
    #     azim_num=30, elev_num=30, geom_type=geom_type)

    # Check output
    print("Range of values [min, max]:")
    print("Sky view factor: %.4f" % np.nanmin(sky_view_factor)
          + ", %.4f" % np.nanmax(sky_view_factor))
    print("Area increase factor: %.4f" % np.nanmin(area_increase_factor)
          + ", %.4f" % np.nanmax(area_increase_factor))
    print("Sky view area factor: %.4f" % np.nanmin(sky_view_area_factor)
          + ", %.4f" % np.nanmax(sky_view_area_factor))
    print("Spatial mean of sky view area factor: %.4f"
          % np.nanmean(sky_view_area_factor))
    print("Surface slope: %.4f" % np.nanmin(slope)
          + ", %.4f" % np.nanmax(slope))
    print("Surface aspect: %.4f" % np.nanmin(aspect)
          + ", %.4f" % np.nanmax(aspect))

    # Save to NetCDF file
    file_out = "Sky_view_factor_" + "_".join(i.split("/")[-1].split("_")[2:])
    ncfile = Dataset(filename=path_work + file_out, mode="w")
    # -------------------------------------------------------------------------
    ncfile.pixel_per_gc = str(pixel_per_gc)
    ncfile.offset_gc = str(offset_gc)
    ncfile.dist_search = "%.2f" % dist_search + " km"
    ncfile.geom_type = geom_type
    ncfile.hori_azim_num = str(hori_azim_num)
    ncfile.hori_acc = "%.2f" % hori_acc
    # -------------------------------------------------------------------------
    nc_meta = ncfile.createVariable("rotated_pole", "S1", )
    nc_meta.grid_mapping_name = "rotated_latitude_longitude"
    nc_meta.grid_north_pole_longitude = pole_lon
    nc_meta.grid_north_pole_latitude = pole_lat
    nc_meta.north_pole_grid_longitude = 0.0
    # -------------------------------------------------------------------------
    ncfile.createDimension(dimname="rlat_gc", size=sky_view_factor.shape[0])
    ncfile.createDimension(dimname="rlon_gc", size=sky_view_factor.shape[1])
    # -------------------------------------------------------------------------
    nc_rlat = ncfile.createVariable(varname="rlat_gc", datatype="f8",
                                    dimensions="rlat_gc")
    nc_rlat[:] = rlat_gc[offset_gc:-offset_gc]
    nc_rlat.long_name = "latitude of grid cells in rotated pole grid"
    nc_rlat.units = "degrees"
    nc_rlon = ncfile.createVariable(varname="rlon_gc", datatype="f8",
                                    dimensions="rlon_gc")
    nc_rlon[:] = rlon_gc[offset_gc:-offset_gc]
    nc_rlon.long_name = "longitude of grid cells in rotated pole grid"
    nc_rlon.units = "degrees"
    # -------------------------------------------------------------------------
    nc_data = ncfile.createVariable(varname="sky_view_factor", datatype="f4",
                                    dimensions=("rlat_gc", "rlon_gc"))
    nc_data[:] = sky_view_factor.astype(np.float32)
    nc_data.units = "-"
    nc_data = ncfile.createVariable(varname="area_increase_factor",
                                    datatype="f4",
                                    dimensions=("rlat_gc", "rlon_gc"))
    nc_data[:] = area_increase_factor.astype(np.float32)
    nc_data.units = "-"
    nc_data = ncfile.createVariable(varname="sky_view_area_factor",
                                    datatype="f4",
                                    dimensions=("rlat_gc", "rlon_gc"))
    nc_data[:] = sky_view_area_factor.astype(np.float32)
    nc_data.units = "-"
    # -------------------------------------------------------------------------
    nc_data = ncfile.createVariable(varname="slope",
                                    datatype="f4",
                                    dimensions=("rlat_gc", "rlon_gc"))
    nc_data[:] = slope.astype(np.float32)
    nc_data.units = "degree"
    nc_data = ncfile.createVariable(varname="aspect",
                                    datatype="f4",
                                    dimensions=("rlat_gc", "rlon_gc"))
    nc_data[:] = aspect.astype(np.float32)
    nc_data.units = "degree"
    # -------------------------------------------------------------------------
    # nc_data = ncfile.createVariable(varname="distance",
    #                                 datatype="f4",
    #                                 dimensions=("rlat_gc", "rlon_gc"))
    # nc_data[:] = distance.astype(np.float32)
    # nc_data.units = "m"
    # -------------------------------------------------------------------------
    ncfile.close()

# -----------------------------------------------------------------------------
# Merge sub-domains (if required)
# -----------------------------------------------------------------------------

if len(files_in) > 1:

    print("Spatially merge sub-domain output files")
    files_out = glob.glob(path_work + "Sky_view_factor_"
                          + "_".join(file_in.split("_")[2:]))
    ds = xr.open_mfdataset(files_out)
    print(np.abs(np.diff(ds["rlon_gc"].values) - float(file_in[21:26])).max())
    print(np.abs(np.diff(ds["rlat_gc"].values) - float(file_in[21:26])).max())
    ds.to_netcdf(files_out[0][:-9] + ".nc")
    ds.close()

    time.sleep(1)
    for i in files_out:
        os.remove(i)
