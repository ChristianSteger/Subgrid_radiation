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
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import time
import subgrid_radiation as subrad
import subgrid_radiation.ocean_masking as ocean_masking
from utilities.grid import coord_edges, grid_frame

mpl.style.use("classic")

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
path_work = "/Users/csteger/Desktop/dir_work/"  # working directory
file_out = "SW_dir_cor_lookup.nc"

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
print("Load data")

# Load data
ds = xr.open_dataset(path_work + "MERIT_remapped_COSMO_0.11deg_y1_x1.nc")
num_gc_in_x = ds.attrs["num_grid_cells_inner_zonal"]
num_gc_in_y = ds.attrs["num_grid_cells_inner_meridional"]
# number of grid cells in inner domain
pixel_per_gc_x = ds.attrs["pixels_per_grid_cell_zonal"]
pixel_per_gc_y = ds.attrs["pixels_per_grid_cell_meridional"]
# pixel per grid cell (along one dimension)
offset_gc_x = ds.attrs["offset_grid_cells_zonal"]
offset_gc_y = ds.attrs["offset_grid_cells_meridional"]
# offset in number of grid cells
# lon = ds["lon"].values.astype(np.float64)
# lat = ds["lat"].values.astype(np.float64)
# elevation = ds["Elevation"].values
mask_water = ds["mask_water"].values.astype(bool)  # (water: True, land: False)
pole_lon = ds["rotated_pole"].grid_north_pole_longitude
pole_lat = ds["rotated_pole"].grid_north_pole_latitude
rlon = ds["rlon"].values
rlat = ds["rlat"].values
rlon_gc = ds["rlon_gc"].values
rlat_gc = ds["rlat_gc"].values
pole_longitude = ds["rotated_pole"].grid_north_pole_longitude
pole_latitude = ds["rotated_pole"].grid_north_pole_latitude
ds.close()

# -----------------------------------------------------------------------------
# Mask water grid cells with a certain minimal distance from coastline
# -----------------------------------------------------------------------------

# Compute contour lines of coast
t_beg = time.perf_counter()
mask_bin = (~mask_water).astype(np.uint8)  # (0: water, 1: land)
contours_rlatrlon = ocean_masking.coastline_contours(rlon, rlat, mask_bin)
print("Run time: %.2f" % (time.perf_counter() - t_beg) + " s")

# Compute minimal chord distance to coastline for water grid cells
dist_chord = ocean_masking.coastline_distance(
    contours_rlatrlon, mask_water, rlon, rlat,
    pixel_per_gc_x, pixel_per_gc_y)

# Plot preparations
ccrs_rot_pole = ccrs.RotatedPole(pole_latitude=pole_latitude,
                                 pole_longitude=pole_longitude,
                                 globe=ccrs.Globe(datum="WGS84",
                                                  ellipse="sphere"))
cmap = plt.get_cmap("YlGnBu")
levels = np.arange(0.0, 550.0, 50.0)
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="max")
rlon_fr, rlat_fr = grid_frame(*coord_edges(rlon_gc, rlat_gc),
                              offset=offset_gc_x)

# Overview plot
fig = plt.figure(figsize=(12.0, 9.5))
ax = plt.axes(projection=ccrs_rot_pole)
ax.set_facecolor("lightgrey")
plt.pcolormesh(rlon_gc, rlat_gc, (dist_chord / 1000.0), shading="auto",
               cmap=cmap, norm=norm)
plt.colorbar()
plt.contour(rlon_gc, rlat_gc, (dist_chord / 1000.0), levels=[dist_search, ],
            colors="black", linewidths=2.5)
for i in contours_rlatrlon:
    plt.plot(i[:, 0], i[:, 1], color="black")
poly = plt.Polygon(list(zip(rlon_fr, rlat_fr)), facecolor="none",
                   edgecolor="black", linestyle="--")
ax.add_patch(poly)
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.5, color="black",
                  alpha=0.8, linestyle="-", draw_labels=True,
                  x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False
d_rlon_h = np.diff(rlon_gc).mean() / 2.0
d_rlat_h = np.diff(rlat_gc).mean() / 2.0
plt.axis([rlon_gc[0] - d_rlon_h, rlon_gc[-1] + d_rlon_h,
          rlat_gc[0] - d_rlat_h, rlat_gc[-1] + d_rlat_h])
plt.title("Minimal chhord distance from coastline [km]",
          fontsize=12, fontweight="bold", y=1.01, loc="left")
temp = dist_chord[offset_gc_y:-offset_gc_y, offset_gc_x:-offset_gc_x]
frac_mask = (temp > (dist_search * 1000.0)).sum() / temp.size
plt.title("Masked grid cells: %.1f" % (frac_mask * 100.0) + " %",
          fontsize=11, y=1.01, loc="right")
fig.savefig(path_work + "ocean_grid_cell_masking.png", dpi=500,
            bbox_inches="tight")
plt.close(fig)

# Mask for grid cells
mask = ~(dist_chord[offset_gc_y:-offset_gc_y, offset_gc_x:-offset_gc_x]
         > (dist_search * 1000.0))
plt.figure()
plt.pcolormesh(mask)

# -----------------------------------------------------------------------------
# Coordinate transformation
# -----------------------------------------------------------------------------
print("Coordinate transformation")


######### continue...



# Transform elevation data (geographic/geodetic -> ENU coordinates)
x_ecef, y_ecef, z_ecef = subrad.transform.lonlat2ecef(lon, lat, elevation,
                                                    ellps="sphere")
dem_dim_0, dem_dim_1 = elevation.shape
trans_ecef2enu = subrad.transform.TransformerEcef2enu(
    lon_or=lon.mean(), lat_or=lat.mean(), ellps="sphere")
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
                                 elevation_zero[slice_in], ellps="sphere")
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
                                 subsol_dist_2d, ellps="sphere")
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
ncfile = Dataset(filename=path_work + file_out, mode="w")
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
