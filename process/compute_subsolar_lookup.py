# Description: Computation of subgrid correction factors for direct downward
#              shortwave radiation (aggregated to the model grid cell
#              resolution) for an array of sun positions
#
# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import os
import numpy as np
import xarray as xr
from skyfield.api import Distance
from netCDF4 import Dataset
import subgrid_radiation as subrad

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# Grid for subsolar points
# subsol_lon = np.linspace(-180.0, 162.0, 10, dtype=np.float64)  # 38 degree
# subsol_lat = np.linspace(-23.5, 23.5, 5, dtype=np.float64)  # 11.75 degree
# subsol_lon = np.linspace(-180.0, 172.0, 45, dtype=np.float64)  # 8 degree
# subsol_lat = np.linspace(-23.5, 23.5, 15, dtype=np.float64)  # 3.36 degree
subsol_lon = np.linspace(-180.0, 175.0, 72, dtype=np.float64)  # 5 degree
subsol_lat = np.linspace(-23.5, 23.5, 21, dtype=np.float64)  # 2.35 degree

# Ray-tracing and 'SW_dir_cor' calculation
dist_search = 100.0  # search distance for terrain shading [kilometre]
geom_type = "grid"  # "grid" or "quad"
ang_max = 89.5
sw_dir_cor_max = 20.0

# Miscellaneous settings
dir_work = "/Users/csteger/Desktop/dir_work/"  # working directory
file_out = "SW_dir_cor_lookup.nc"
ellps = "sphere"  # Earth's surface approximation (sphere, GRS80 or WGS84)

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
print("Load data")

# Load data
ds = xr.open_dataset(dir_work + "MERIT_remapped_COSMO.nc")
pixel_per_gc = ds.attrs["sub_grid_info_zonal"]
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

# -----------------------------------------------------------------------------
# Coordinate transformation
# -----------------------------------------------------------------------------
print("Coordinate transformation")

# Transform elevation data (geographic/geodetic -> ENU coordinates)
x_ecef, y_ecef, z_ecef = subrad.transform.lonlat2ecef(lon, lat, elevation,
                                                    ellps=ellps)
dem_dim_0, dem_dim_1 = elevation.shape
trans_ecef2enu = subrad.transform.TransformerEcef2enu(
    lon_or=lon.mean(), lat_or=lat.mean(), ellps=ellps)
x_enu, y_enu, z_enu = subrad.transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                              trans_ecef2enu)
del x_ecef, y_ecef, z_ecef

# Merge vertex coordinates and pad geometry buffer
vert_grid = subrad.auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)
print("Size of elevation data: %.3f" % (vert_grid.nbytes / (10 ** 9))
      + " GB")
del x_enu, y_enu, z_enu

# Transform 0.0 m surface data (geographic/geodetic -> ENU coordinates)
slice_in = (slice(pixel_per_gc * offset_gc, -pixel_per_gc * offset_gc),
            slice(pixel_per_gc * offset_gc, -pixel_per_gc * offset_gc))
elevation_zero = np.zeros_like(elevation)
x_ecef, y_ecef, z_ecef \
    = subrad.transform.lonlat2ecef(lon[slice_in], lat[slice_in],
                                 elevation_zero[slice_in], ellps=ellps)
dem_dim_in_0, dem_dim_in_1 = elevation_zero[slice_in].shape
x_enu, y_enu, z_enu = subrad.transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                              trans_ecef2enu)
del x_ecef, y_ecef, z_ecef

# Merge vertex coordinates and pad geometry buffer
vert_grid_in = subrad.auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)
print("Size of elevation data (0.0 m surface): %.3f"
      % (vert_grid_in.nbytes / (10 ** 9)) + " GB")
del x_enu, y_enu, z_enu

# Transform locations of subsolar points
subsol_lon_2d, subsol_lat_2d = np.meshgrid(subsol_lon, subsol_lat)
subsol_dist_2d = np.empty(subsol_lon_2d.shape, dtype=np.float32)
subsol_dist_2d[:] = Distance(au=1).m
# astronomical unit (~average Sun-Earth distance) [m]
x_ecef, y_ecef, z_ecef \
    = subrad.transform.lonlat2ecef(subsol_lon_2d, subsol_lat_2d,
                                 subsol_dist_2d, ellps=ellps)
x_enu, y_enu, z_enu = subrad.transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                              trans_ecef2enu)

# Combine sun position in one array
sun_pos = np.concatenate((x_enu[:, :, np.newaxis],
                          y_enu[:, :, np.newaxis],
                          z_enu[:, :, np.newaxis]), axis=2)

# -----------------------------------------------------------------------------
# Compute spatially aggregated correction factors
# -----------------------------------------------------------------------------
print("Compute spatially aggregated correction factors")

# Compute
# sw_dir_cor = subrad.subsolar_lookup.sw_dir_cor(
# sw_dir_cor = subrad.subsolar_lookup.sw_dir_cor_coherent(
sw_dir_cor = subrad.subsolar_lookup.sw_dir_cor_coherent_rp8(
    vert_grid, dem_dim_0, dem_dim_1,
    vert_grid_in, dem_dim_in_0, dem_dim_in_1,
    sun_pos, pixel_per_gc, offset_gc,
    dist_search=dist_search, geom_type=geom_type,
    ang_max=ang_max, sw_dir_cor_max=sw_dir_cor_max)

# Check output
print("Range of 'sw_dir_cor'-values: [%.2f" % sw_dir_cor.min()
      + ", %.2f" % sw_dir_cor.max() + "]")
print("Size of lookup table: %.2f" % (sw_dir_cor.nbytes / (10 ** 6)) + " MB")

# Save to NetCDF file
ncfile = Dataset(filename=dir_work + file_out, mode="w")
ncfile.pixel_per_gc = str(pixel_per_gc)
ncfile.offset_gc = str(offset_gc)
ncfile.dist_search = "%.2f" % dist_search + " km"
ncfile.geom_type = geom_type
ncfile.ang_max = "%.2f" % ang_max + " degrees"
ncfile.sw_dir_cor_max = "%.2f" % sw_dir_cor_max
# -----------------------------------------------------------------------------
ncfile.createDimension(dimname="gc_lat", size=sw_dir_cor.shape[0])
ncfile.createDimension(dimname="gc_lon", size=sw_dir_cor.shape[1])
ncfile.createDimension(dimname="subsolar_lat", size=sw_dir_cor.shape[2])
ncfile.createDimension(dimname="subsolar_lon", size=sw_dir_cor.shape[3])
# -----------------------------------------------------------------------------
nc_sslat = ncfile.createVariable(varname="subsolar_lat", datatype="f",
                                dimensions="subsolar_lat")
nc_sslat[:] = subsol_lat
nc_sslat.long_name = "subsolar latitude"
nc_sslat.units = "degrees"
nc_sslon = ncfile.createVariable(varname="subsolar_lon", datatype="f",
                                dimensions="subsolar_lon")
nc_sslon[:] = subsol_lon
nc_sslon.long_name = "subsolar longitude"
nc_sslon.units = "degrees"
# -----------------------------------------------------------------------------
nc_data = ncfile.createVariable(varname="f_cor", datatype="f",
                                dimensions=("gc_lat", "gc_lon",
                                            "subsolar_lat", "subsolar_lon"))
nc_data[:] = sw_dir_cor
nc_data.units = "-"
# -----------------------------------------------------------------------------
ncfile.close()
