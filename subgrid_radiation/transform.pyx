#cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np
from libc.math cimport sin, cos
from libc.math cimport M_PI
from cython.parallel import prange


# -----------------------------------------------------------------------------

def lonlat2ecef(lon, lat, h, trans_lonlat2enu, in_place=False):
    """Transformation of spherical longitude/latitude to earth-centered,
    earth-fixed (ECEF) coordinates.

    Parameters
    ----------
    lon : ndarray of double
        Array (with arbitrary dimensions) with geographic longitude [degree]
    lat : ndarray of double
        Array (with arbitrary dimensions) with geographic latitude [degree]
    h : ndarray of double
        Array (with arbitrary dimensions) with elevation above sphere [metre]
    trans_lonlat2enu : class
        Instance of class `TransformerLonlat2enu`
    in_place : bool
        Option to perform transformation in-place
        (x_ecef -> lon, y_ecef -> lat, z_ecef -> h)

    Returns (optional)
    -------
    x_ecef : ndarray of double
        Array (dimensions according to input) with ECEF x-coordinates [metre]
    y_ecef : ndarray of double
        Array (dimensions according to input) with ECEF y-coordinates [metre]
    z_ecef : ndarray of double
        Array (dimensions according to input) with ECEF z-coordinates [metre]
        """

    # Check arguments
    if (lon.shape != lat.shape) or (lat.shape != h.shape):
        raise ValueError("Inconsistent shapes / number of dimensions of "
                         + "input arrays")
    if ((lon.dtype != "float64") or (lat.dtype != "float64")
            or (h.dtype != "float64")):
        raise ValueError("Input array(s) has/have incorrect data type(s)")
    if not isinstance(trans_lonlat2enu, TransformerLonlat2enu):
        raise ValueError("Last input argument must be instance of class "
                         + "'TransformerLonlat2enu'")

    # Wrapper for 1-dimensional function
    shp = lon.shape
    if not in_place:
        x_ecef, y_ecef, z_ecef = _lonlat2ecef_1d(lon.ravel(), lat.ravel(),
                                                  h.ravel(), trans_lonlat2enu)
        return x_ecef.reshape(shp), y_ecef.reshape(shp), z_ecef.reshape(shp)
    else:
        _lonlat2ecef_1d_in_place(lon.ravel(), lat.ravel(), h.ravel(),
                                 trans_lonlat2enu)


def _lonlat2ecef_1d(double[:] lon, double[:] lat, double[:] h,
                    trans_lonlat2enu):
    """Coordinate transformation from lon/lat to ECEF (for 1-dimensional data).

    Sources
    -------
    - https://en.wikipedia.org/wiki/Geographic_coordinate_conversion"""

    cdef int len_0 = lon.shape[0]
    cdef int i
    cdef double r, f, a, b, e_2, n
    cdef double radius_earth = trans_lonlat2enu.radius_earth
    cdef double[:] x_ecef = np.empty(len_0, dtype=np.float64)
    cdef double[:] y_ecef = np.empty(len_0, dtype=np.float64)
    cdef double[:] z_ecef = np.empty(len_0, dtype=np.float64)

    for i in prange(len_0, nogil=True, schedule="static"):
        x_ecef[i] = (radius_earth + h[i]) * cos(deg2rad(lat[i])) \
                    * cos(deg2rad(lon[i]))
        y_ecef[i] = (radius_earth + h[i]) * cos(deg2rad(lat[i])) \
                    * sin(deg2rad(lon[i]))
        z_ecef[i] = (radius_earth + h[i]) * sin(deg2rad(lat[i]))

    return np.asarray(x_ecef), np.asarray(y_ecef), np.asarray(z_ecef)


def _lonlat2ecef_1d_in_place(double[:] lon, double[:] lat, double[:] h,
                             trans_lonlat2enu):
    """Coordinate transformation from lon/lat to ECEF (for 1-dimensional data).

    Sources
    -------
    - https://en.wikipedia.org/wiki/Geographic_coordinate_conversion"""

    cdef int len_0 = lon.shape[0]
    cdef int i
    cdef double r, f, a, b, e_2, n
    cdef double lon_temp, lat_temp, h_temp
    cdef double radius_earth = trans_lonlat2enu.radius_earth

    for i in prange(len_0, nogil=True, schedule="static"):
        lon_temp = deg2rad(lon[i])
        lat_temp = deg2rad(lat[i])
        h_temp = h[i]
        lon[i] = (radius_earth + h_temp) * cos(lat_temp) * cos(lon_temp)
        lat[i] = (radius_earth + h_temp) * cos(lat_temp) * sin(lon_temp)
        h[i] = (radius_earth + h_temp) * sin(lat_temp)


# -----------------------------------------------------------------------------

def ecef2enu(x_ecef, y_ecef, z_ecef, trans_lonlat2enu, in_place=False):
    """Coordinate transformation from ECEF to ENU.

    Transformation of earth-centered, earth-fixed (ECEF) to local tangent
    plane (ENU) coordinates.

    Parameters
    ----------
    x_ecef : ndarray of double
        Array (with arbitrary dimensions) with ECEF x-coordinates [metre]
    y_ecef : ndarray of double
        Array (with arbitrary dimensions) with ECEF y-coordinates [metre]
    z_ecef : ndarray of double
        Array (with arbitrary dimensions) with ECEF z-coordinates [metre]
    trans_lonlat2enu : class
        Instance of class `TransformerLonlat2enu`
    in_place : bool
        Option to perform transformation in-place
        (x_enu -> x_ecef, y_enu -> y_ecef, z_enu -> z_ecef)

    Returns
    -------
    x_enu : ndarray of double
        Array (dimensions according to input) with ENU x-coordinates [metre]
    y_enu : ndarray of double
        Array (dimensions according to input) with ENU y-coordinates [metre]
    z_enu : ndarray of double
        Array (dimensions according to input) with ENU z-coordinates [metre]"""

    # Check arguments
    if (x_ecef.shape != y_ecef.shape) or (y_ecef.shape != z_ecef.shape):
        raise ValueError("Inconsistent shapes / number of dimensions of "
                         + "input arrays")
    if ((x_ecef.dtype != "float64") or (y_ecef.dtype != "float64")
            or (z_ecef.dtype != "float64")):
        raise ValueError("Input array(s) has/have incorrect data type(s)")
    if not isinstance(trans_lonlat2enu, TransformerLonlat2enu):
        raise ValueError("Last input argument must be instance of class "
                         + "'TransformerLonlat2enu'")

    # Wrapper for 1-dimensional function
    shp = x_ecef.shape
    if not in_place:
        x_enu, y_enu, z_enu = _ecef2enu_1d(x_ecef.ravel(), y_ecef.ravel(),
                                           z_ecef.ravel(),
                                           trans_lonlat2enu)
        return x_enu.reshape(shp), y_enu.reshape(shp), z_enu.reshape(shp)
    else:
        _ecef2enu_1d_in_place(x_ecef.ravel(), y_ecef.ravel(), z_ecef.ravel(),
                              trans_lonlat2enu)


def _ecef2enu_1d(double[:] x_ecef, double[:] y_ecef, double[:] z_ecef,
                trans_lonlat2enu):
    """Coordinate transformation from ECEF to ENU (for 1-dimensional data).

    Sources
    -------
    - https://en.wikipedia.org/wiki/Geographic_coordinate_conversion"""

    cdef int len_0 = x_ecef.shape[0]
    cdef int i
    cdef double sin_lon, cos_lon, sin_lat, cos_lat
    cdef double x_ecef_or = trans_lonlat2enu.x_ecef_or
    cdef double y_ecef_or = trans_lonlat2enu.y_ecef_or
    cdef double z_ecef_or = trans_lonlat2enu.z_ecef_or
    cdef double lon_or = trans_lonlat2enu.lon_or
    cdef double lat_or = trans_lonlat2enu.lat_or
    cdef double[:] x_enu = np.empty(len_0, dtype=np.float64)
    cdef double[:] y_enu = np.empty(len_0, dtype=np.float64)
    cdef double[:] z_enu = np.empty(len_0, dtype=np.float64)

    # Trigonometric functions
    sin_lon = sin(deg2rad(lon_or))
    cos_lon = cos(deg2rad(lon_or))
    sin_lat = sin(deg2rad(lat_or))
    cos_lat = cos(deg2rad(lat_or))

    # Coordinate transformation
    for i in prange(len_0, nogil=True, schedule="static"):
        x_enu[i] = (- sin_lon * (x_ecef[i] - x_ecef_or)
                    + cos_lon * (y_ecef[i] - y_ecef_or))
        y_enu[i] = (- sin_lat * cos_lon * (x_ecef[i] - x_ecef_or)
                    - sin_lat * sin_lon * (y_ecef[i] - y_ecef_or)
                    + cos_lat * (z_ecef[i] - z_ecef_or))
        z_enu[i] = (+ cos_lat * cos_lon * (x_ecef[i] - x_ecef_or)
                    + cos_lat * sin_lon * (y_ecef[i] - y_ecef_or)
                    + sin_lat * (z_ecef[i] - z_ecef_or))

    return np.asarray(x_enu), np.asarray(y_enu), np.asarray(z_enu)


def _ecef2enu_1d_in_place(double[:] x_ecef, double[:] y_ecef, double[:] z_ecef,
                          trans_lonlat2enu):
    """Coordinate transformation from ECEF to ENU (for 1-dimensional data).

    Sources
    -------
    - https://en.wikipedia.org/wiki/Geographic_coordinate_conversion"""

    cdef int len_0 = x_ecef.shape[0]
    cdef int i
    cdef double sin_lon, cos_lon, sin_lat, cos_lat
    cdef double x_ecef_temp, y_ecef_temp, z_ecef_temp
    cdef double x_ecef_or = trans_lonlat2enu.x_ecef_or
    cdef double y_ecef_or = trans_lonlat2enu.y_ecef_or
    cdef double z_ecef_or = trans_lonlat2enu.z_ecef_or
    cdef double lon_or = trans_lonlat2enu.lon_or
    cdef double lat_or = trans_lonlat2enu.lat_or

    # Trigonometric functions
    sin_lon = sin(deg2rad(lon_or))
    cos_lon = cos(deg2rad(lon_or))
    sin_lat = sin(deg2rad(lat_or))
    cos_lat = cos(deg2rad(lat_or))

    # Coordinate transformation
    for i in prange(len_0, nogil=True, schedule="static"):
        x_ecef_temp = x_ecef[i]
        y_ecef_temp = y_ecef[i]
        z_ecef_temp = z_ecef[i]
        x_ecef[i] = (- sin_lon * (x_ecef_temp - x_ecef_or)
                     + cos_lon * (y_ecef_temp - y_ecef_or))
        y_ecef[i] = (- sin_lat * cos_lon * (x_ecef_temp - x_ecef_or)
                     - sin_lat * sin_lon * (y_ecef_temp - y_ecef_or)
                     + cos_lat * (z_ecef_temp - z_ecef_or))
        z_ecef[i] = (+ cos_lat * cos_lon * (x_ecef_temp - x_ecef_or)
                     + cos_lat * sin_lon * (y_ecef_temp - y_ecef_or)
                     + sin_lat * (z_ecef_temp - z_ecef_or))


# -----------------------------------------------------------------------------

class TransformerLonlat2enu:
    """Class that stores attributes to transform from ECEF to ENU coordinates.

    Transformer class that stores attributes to convert between ECEF and ENU
    coordinates. The origin of the ENU coordinate system coincides with the
    surface of the sphere.

    Parameters
    -------
    lon_or : double
        Longitude coordinate for origin of ENU coordinate system [degree]
    lat_or : double
        Latitude coordinate for origin of ENU coordinate system [degree]
    radius_earth : double
        Radius of Earth [metre]"""

    def __init__(self, lon_or, lat_or, radius_earth):
        if (lon_or < -180.0) or (lon_or > 180.0):
            raise ValueError("Value for 'lon_or' is outside of valid range")
        if (lat_or < -90.0) or (lat_or > 90.0):
            raise ValueError("Value for 'lat_or' is outside of valid range")
        self.lon_or = lon_or
        self.lat_or = lat_or
        self.radius_earth = radius_earth

        # Origin of ENU coordinate system in ECEF coordinates
        self.x_ecef_or = self.radius_earth * np.cos(np.deg2rad(self.lat_or)) \
                         * np.cos(np.deg2rad(self.lon_or))
        self.y_ecef_or = self.radius_earth * np.cos(np.deg2rad(self.lat_or)) \
                         * np.sin(np.deg2rad(self.lon_or))
        self.z_ecef_or = self.radius_earth * np.sin(np.deg2rad(self.lat_or))

        # North Pole in ENU coordinate system
        self.x_enu_np = (- np.sin(np.deg2rad(self.lon_or))
                         * (0.0 - self.x_ecef_or)
                         + np.cos(np.deg2rad(self.lon_or))
                         * (0.0 - self.y_ecef_or))
        self.y_enu_np = (- np.sin(np.deg2rad(self.lat_or))
                         * np.cos(np.deg2rad(self.lon_or))
                         * (0.0 - self.x_ecef_or)
                         - np.sin(np.deg2rad(self.lat_or))
                         * np.sin(np.deg2rad(self.lon_or))
                         * (0.0 - self.y_ecef_or)
                         + np.cos(np.deg2rad(self.lat_or))
                         * (self.radius_earth - self.z_ecef_or))
        self.z_enu_np = (+ np.cos(np.deg2rad(self.lat_or))
                         * np.cos(np.deg2rad(self.lon_or))
                         * (0.0 - self.x_ecef_or)
                         + np.cos(np.deg2rad(self.lat_or))
                         * np.sin(np.deg2rad(self.lon_or))
                         * (0.0 - self.y_ecef_or)
                         + np.sin(np.deg2rad(self.lat_or))
                         * (self.radius_earth - self.z_ecef_or))

# -----------------------------------------------------------------------------
# Auxiliary function(s)
# -----------------------------------------------------------------------------

cdef inline double deg2rad(double ang_in) nogil:
    """Convert degree to radian"""

    cdef double ang_out
    
    ang_out = ang_in * (M_PI / 180.0)
       
    return ang_out
