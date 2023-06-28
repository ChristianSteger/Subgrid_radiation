# Description: Test coordinate transformation
#
# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np
import time
from subgrid_radiation import transform
from pyproj import CRS, Transformer

# -----------------------------------------------------------------------------
# Check correct transformation
# -----------------------------------------------------------------------------

# Reference location for ENU coordinates, earth radius
lon_or = -164.7
lat_or = -33.4
radius_earth = 6378137.0  # PROJ [m]

# Test data
num = 10_000
lon = np.random.uniform(-180.0, 180.0, num)
lat = np.random.uniform(-90.0, 90.0, num)
elevation = np.random.uniform(0.0, 4500.0, num)

# Spherical latitude/longitude to ECEF
trans_ecef2enu = transform.TransformerEcef2enu(
    lon_or=lon_or, lat_or=lat_or, radius_earth=radius_earth)
out = transform.lonlat2ecef(lon, lat, elevation, trans_ecef2enu)
crs_s = CRS.from_json_dict({"proj": "latlong", "R": radius_earth})
crs_t = CRS.from_json_dict({"proj": "geocent", "R": radius_earth})
transformer = Transformer.from_crs(crs_s, crs_t)
out_pp = transformer.transform(lon, lat, elevation, radians=False)
print(np.abs(out[0] - out_pp[0]).max())
print(np.abs(out[1] - out_pp[1]).max())
print(np.abs(out[2] - out_pp[2]).max())

# ECEF to ENU
out = transform.ecef2enu(out_pp[0], out_pp[1], out_pp[2], trans_ecef2enu)
ecef2enu = Transformer.from_pipeline(
    "+proj=pipeline +step +proj=cart +R=6378137.0 +step +proj=topocentric "
    + "+R=6378137.0 +lon_0=" + str(lon_or) + " +lat_0=" + str(lat_or)
    + " +h_0=0.0")
out_pp = ecef2enu.transform(lon, lat, elevation)
print(np.abs(out[0] - out_pp[0]).max())
print(np.abs(out[1] - out_pp[1]).max())
print(np.abs(out[2] - out_pp[2]).max())

# -----------------------------------------------------------------------------
# Test equalitiy and preformance of in-place operation
# -----------------------------------------------------------------------------

# Test data
shp = (10_000, 10_000)
lon = np.linspace(0.0, 180.0, shp[0] * shp[0], dtype=np.float64).reshape(shp)
lat = np.linspace(-45.0, 45.0, shp[0] * shp[0], dtype=np.float64).reshape(shp)
elevation = np.linspace(0.0, 4500.0, shp[0] * shp[0],
                        dtype=np.float64).reshape(shp)
radius_earth = 6378137.0  # PROJ [m]

# Transform elevation data (geographic/geodetic -> ENU coordinates)
trans_ecef2enu = transform.TransformerEcef2enu(
    lon_or=lon.mean(), lat_or=lat.mean(), radius_earth=radius_earth)

x_ecef, y_ecef, z_ecef = transform.lonlat2ecef(lon, lat, elevation,
                                              trans_ecef2enu)
x_enu, y_enu, z_enu = transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                         trans_ecef2enu)

transform.lonlat2ecef(lon, lat, elevation, trans_ecef2enu, in_place=True)
print(np.abs(x_ecef - lon).max())
print(np.abs(y_ecef - lat).max())
print(np.abs(z_ecef - elevation).max())
transform.ecef2enu(lon, lat, elevation, trans_ecef2enu, in_place=True)
print(np.abs(x_enu - lon).max())
print(np.abs(y_enu - lat).max())
print(np.abs(z_enu - elevation).max())

# Default transformation
lon = np.linspace(0.0, 180.0, shp[0] * shp[0], dtype=np.float64).reshape(shp)
lat = np.linspace(-45.0, 45.0, shp[0] * shp[0], dtype=np.float64).reshape(shp)
elevation = np.linspace(0.0, 4500.0, shp[0] * shp[0],
                        dtype=np.float64).reshape(shp)
t_beg = time.perf_counter()
x_ecef, y_ecef, z_ecef = transform.lonlat2ecef(lon, lat, elevation,
                                              trans_ecef2enu)
x_enu, y_enu, z_enu = transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                         trans_ecef2enu)
print("Elapsed time: %.3f" % (time.perf_counter() - t_beg) + " s")
del x_ecef, y_ecef, z_ecef, x_enu, y_enu, z_enu

# Inplace transformation
lon = np.linspace(0.0, 180.0, shp[0] * shp[0], dtype=np.float64).reshape(shp)
lat = np.linspace(-45.0, 45.0, shp[0] * shp[0], dtype=np.float64).reshape(shp)
elevation = np.linspace(0.0, 4500.0, shp[0] * shp[0],
                        dtype=np.float64).reshape(shp)
t_beg = time.perf_counter()
transform.lonlat2ecef(lon, lat, elevation, trans_ecef2enu, in_place=True)
transform.ecef2enu(lon, lat, elevation, trans_ecef2enu, in_place=True)
print("Elapsed time: %.3f" % (time.perf_counter() - t_beg) + " s")
