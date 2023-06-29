# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

# Load modules
import numpy as np
from scipy.spatial import cKDTree
import time
from skimage.measure import find_contours
import subgrid_radiation.transform as transform


# -----------------------------------------------------------------------------

def coastline_contours(rlon, rlat, mask_bin):
    """Compute coastline contours from binary land-sea mask.

    Parameters
    ----------
    rlon : ndarray of double
        Array (1-dimensional) with rotated longitude [degree]
    rlat: ndarray of double
        Array (1-dimensional) with rotated latitude [degree]
    mask_bin: ndarray of 8-bit unsigned integer
        Array (2-dimensional) with binary land-sea mask (0: water, 1: land)

    Returns
    -------
    contours_rlatrlon : list of ndarray
        List with contour lines in rotated latitude/longitude coordinates
        [degree]"""

    # Check arguments
    if (rlat.ndim != 1) or (rlon.ndim != 1):
        raise ValueError("Input coordinates arrays must be 1-dimensional")
    if (mask_bin.shape[0] != len(rlat)) or (mask_bin.shape[1] != len(rlon)):
        raise ValueError("Input data has inconsistent dimension length(s)")
    if (mask_bin.dtype != "uint8") or (len(np.unique(mask_bin)) != 2) \
            or (not np.all(np.unique(mask_bin) == [0, 1])):
        raise ValueError("'mask_bin' must be of type 'uint8' and may "
                         + "only contain 0 and 1")

    # Compute contour lines
    contours = find_contours(mask_bin, 0.5, fully_connected="high")

    # Get latitude/longitude coordinates of contours
    rlon_ind = np.linspace(rlon[0], rlon[-1], len(rlon) * 2 - 1)
    rlat_ind = np.linspace(rlat[0], rlat[-1], len(rlat) * 2 - 1)
    contours_rlatrlon = []
    for i in contours:
        pts_rlatrlon = np.empty(i.shape, dtype=np.float64)
        pts_rlatrlon[:, 0] = rlon_ind[(i[:, 1] * 2).astype(np.int32)]
        pts_rlatrlon[:, 1] = rlat_ind[(i[:, 0] * 2).astype(np.int32)]
        contours_rlatrlon.append(pts_rlatrlon)

    return contours_rlatrlon


# -----------------------------------------------------------------------------

def coastline_distance(contours_rlatrlon, mask_water, rlon, rlat,
                       pixel_per_gc_x, pixel_per_gc_y, radius_earth):
    """Compute minimal chord distance between all water grid cells (centre)
    and the coastline. Only grid cells that are classified 'entirely water'
    according to the pixel information are considered - the remaing cells
    are considered land.

    Parameters
    ----------
    contours_rlatrlon : list of ndarray
        List with contour lines in rotated latitude/longitude coordinates
        [degree]
    mask_water : ndarray of bool
        Array (two-dimensional) with water pixels (water: True, land: False)
    rlon: ndarray of float
        Array (one-dimensional) with rotated longitude of pixels [degree]
    rlat: ndarray of float
        Array (one-dimensional) with rotated latitude of pixels [degree]
    pixel_per_gc_x: int
        Number of pixels per grid cell in x-direction (zonal)
    pixel_per_gc_y: int
        Number of pixels per grid cell in y-direction (meridional)
    radius_earth : float
        Radius of Earth [metre]

    Returns
    -------
    dist_chord : ndarray of double
        Array (2-dimensional) minimal chord distance between grid cells and
        coastline [metre]"""

    # Check arguments
    if (rlat.ndim != 1) or (rlon.ndim != 1):
        raise ValueError("Input coordinates arrays must be 1-dimensional")
    if (mask_water.shape[0] != len(rlat)) \
            or (mask_water.shape[1] != len(rlon)):
        raise ValueError("Input data has inconsistent dimension length(s)")
    if (mask_water.dtype != "bool"):
        raise ValueError("'mask_water' must be of type 'bool'")

    # Build k-d tree
    contours_rlatrlon_arr = np.vstack(([i for i in contours_rlatrlon]))
    elevation_0 = np.zeros(contours_rlatrlon_arr.shape[0], dtype=np.float64)
    trans_lonlat2enu = transform.TransformerLonlat2enu(
        lon_or=np.nan, lat_or=np.nan, radius_earth=radius_earth)
    x_ecef, y_ecef, z_ecef \
        = transform.lonlat2ecef(contours_rlatrlon_arr[:, 0],
                                contours_rlatrlon_arr[:, 1],
                                elevation_0, trans_lonlat2enu)
    pts_ecef = np.vstack((x_ecef, y_ecef, z_ecef)).transpose()
    tree = cKDTree(pts_ecef)

    # Select values at horizontal grid lines and compute minimal distance
    # to coastline
    t_beg_func = time.time()
    mask_water_gly = mask_water[0:None:pixel_per_gc_y, :]
    rlon_gly = np.repeat(rlon[np.newaxis, :], mask_water_gly.shape[0], axis=0)
    rlat_gly = np.repeat(rlat[0:None:pixel_per_gc_y, np.newaxis],
                         mask_water_gly.shape[1], axis=1)
    x_ecef, y_ecef, z_ecef = transform.lonlat2ecef(
        rlon_gly[mask_water_gly].astype(np.float64),
        rlat_gly[mask_water_gly].astype(np.float64),
        np.zeros(mask_water_gly.sum(), dtype=np.float64), trans_lonlat2enu)
    pts_ecef_gl = np.vstack((x_ecef, y_ecef, z_ecef)).transpose()
    dist_quer, idx = tree.query(pts_ecef_gl, k=1, workers=-1)
    dist_gly = np.empty_like(rlon_gly)
    dist_gly.fill(np.nan)
    dist_gly[mask_water_gly] = dist_quer  # min. distance to coastline [m]

    # Select values at vertical grid lines and compute minimal distance
    # to coastline
    mask_water_glx = mask_water[:, 0:None:pixel_per_gc_x]
    rlon_glx = np.repeat(rlon[np.newaxis, 0:None:pixel_per_gc_x],
                         mask_water_glx.shape[0], axis=0)
    rlat_glx = np.repeat(rlat[:, np.newaxis], mask_water_glx.shape[1], axis=1)
    x_ecef, y_ecef, z_ecef = transform.lonlat2ecef(
        rlon_glx[mask_water_glx].astype(np.float64),
        rlat_glx[mask_water_glx].astype(np.float64),
        np.zeros(mask_water_glx.sum(), dtype=np.float64), trans_lonlat2enu)
    pts_ecef_gl = np.vstack((x_ecef, y_ecef, z_ecef)).transpose()
    dist_quer, idx = tree.query(pts_ecef_gl, k=1, workers=-1)
    dist_glx = np.empty_like(rlon_glx)
    dist_glx.fill(np.nan)
    dist_glx[mask_water_glx] = dist_quer  # min. distance to coastline [m]
    print("Distances along chord lines computed ("
          + "%.1f" % (time.time() - t_beg_func) + " s)")

    # Loop through grid cells
    num_gc_sd_y = mask_water_gly.shape[0] - 1
    num_gc_sd_x = mask_water_glx.shape[1] - 1
    dist_chord = np.zeros((num_gc_sd_y, num_gc_sd_x), dtype=np.float32)
    dist_chord.fill(np.nan)
    for i in range(num_gc_sd_y):
        for j in range(num_gc_sd_x):

            i_pix = i * pixel_per_gc_y
            j_pix = j * pixel_per_gc_x
            water_gly = mask_water[i_pix:(i_pix + pixel_per_gc_y + 1),
                                  j_pix:(j_pix + pixel_per_gc_x + 1)]
            water_glx = mask_water[i_pix:(i_pix + pixel_per_gc_y + 1),
                                  j_pix:(j_pix + pixel_per_gc_x + 1)]
            # -> check all pixels (not only frame) because small islands can
            # lie entirely within grid cells ...
            if (np.all(water_gly) and np.all(water_glx)):

                dist_y = dist_gly[i:(i + 2),
                         j_pix:(j_pix + pixel_per_gc_x + 1)]
                dist_x = dist_glx[i_pix:(i_pix + pixel_per_gc_y + 1),
                         j:(j + 2)]
                if np.any(np.isnan(dist_y)) or np.any(np.isnan(dist_x)):
                    raise ValueError("NaN-value occured")
                dist_chord[i, j] = np.minimum(dist_y.min(), dist_x.min())

    return dist_chord
