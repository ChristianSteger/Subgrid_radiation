# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

cimport numpy as np
import numpy as np
import os

# -----------------------------------------------------------------------------
# Default
# -----------------------------------------------------------------------------

cdef extern from "horizon_comp.h":
    void sw_dir_cor_svf_comp(
            float* vert_grid,
            int dem_dim_0, int dem_dim_1,
            float* vert_grid_in,
            int dem_dim_in_0, int dem_dim_in_1,
            float* sun_pos,
            int dim_sun_0, int dim_sun_1,
            float* sw_dir_cor,
            float* sky_view_factor,
            int pixel_per_gc,
            int offset_gc,
            np.npy_uint8 * mask,
            float dist_search,
            int hori_azim_num,
            char* geom_type,
            float ang_max,
            float sw_dir_cor_max)

def sw_dir_cor_svf(
        np.ndarray[np.float32_t, ndim = 1] vert_grid,
        int dem_dim_0, int dem_dim_1,
        np.ndarray[np.float32_t, ndim = 1] vert_grid_in,
        int dem_dim_in_0, int dem_dim_in_1,
        np.ndarray[np.float32_t, ndim = 3] sun_pos,
        int pixel_per_gc,
        int offset_gc,
        np.ndarray[np.uint8_t, ndim = 2] mask=None,
        float dist_search=100.0,
        int hori_azim_num=72,
        str geom_type="grid",
        float ang_max=89.0,
        float sw_dir_cor_max=25.0):
    """Compute subsolar lookup table of subgrid-scale correction factors
    for direct downward shortwave radiation. Additionally, the sky view factor
    is computed.

    Parameters
    ----------
    vert_grid : ndarray of float
        Array (one-dimensional) with vertices of DEM in ENU coordinates [metre]
    dem_dim_0 : int
        Dimension length of DEM in y-direction
    dem_dim_1 : int
        Dimension length of DEM in x-direction
    vert_grid_in : ndarray of float
        Array (one-dimensional) with vertices of inner DEM with 0.0 m elevation
        in ENU coordinates [metre]
    dem_dim_in_0 : int
        Dimension length of inner DEM in y-direction
    dem_dim_in_1 : int
        Dimension length of inner DEM in x-direction
    sun_pos : ndarray of float
        Array (three-dimensional) with sun positions in ENU coordinates
        (dim_sun_0, dim_sun_1, 3) [metre]
    pixel_per_gc : int
        Number of subgrid pixels within one grid cell (along one dimension)
    offset_gc : int
        Offset number of grid cells
    mask : ndarray of uint8
        Array (two-dimensional) with grid cells for which 'sw_dir_cor' and
        'sky_view_factor' are computed. Masked (0) grid cells are filled with
        NaN.
    dist_search : float
        Search distance for topographic shadowing [kilometre]
    hori_azim_num : int
        Number of azimuth sectors for horizon computation
    geom_type : str
        Embree geometry type (triangle, quad, grid)
    ang_max : float
        Maximal angle between sun vector and tilted surface normal for which
        correction is computed. For larger angles, 'sw_dir_cor' is set to 0.0.
        'ang_max' is also applied to restrict the maximal angle between the sun
        vector and the horizontal surface normal [degree]
    sw_dir_cor_max : float
        Maximal allowed correction factor for direct downward shortwave
        radiation [-]

    Returns
    -------
    sw_dir_cor : ndarray of float
        Array (four-dimensional) with shortwave correction factor
        (y, x, dim_sun_0, dim_sun_1) [-]
    sky_view_factor : ndarray of float
        Array (two-dimensional) with sky view factor (y, x) [-]

    References
    ----------
    - Mueller, M. D., & Scherer, D. (2005): A Grid- and Subgrid-Scale
    Radiation Parameterization of Topographic Effects for Mesoscale
    Weather Forecast Models, Monthly Weather Review, 133(6), 1431-1442."""

	# Check consistency and validity of input arguments
    if ((dem_dim_0 != (2 * offset_gc * pixel_per_gc) + dem_dim_in_0)
            or (dem_dim_1 != (2 * offset_gc * pixel_per_gc) + dem_dim_in_1)):
        raise ValueError("Inconsistency between input arguments 'dem_dim_?',"
                         + " 'dem_dim_in_?', 'offset_gc' and 'pixel_per_gc'")
    if len(vert_grid) < (dem_dim_0 * dem_dim_1 * 3):
        raise ValueError("array 'vert_grid' has insufficient length")
    if len(vert_grid_in) < (dem_dim_in_0 * dem_dim_in_1 * 3):
        raise ValueError("array 'vert_grid_in' has insufficient length")
    if pixel_per_gc < 1:
        raise ValueError("value for 'pixel_per_gc' must be larger than 1")
    if offset_gc < 0:
        raise ValueError("value for 'offset_gc' must be larger than 0")
    num_gc_y = int((dem_dim_0 - 1) / pixel_per_gc) - 2 * offset_gc
    num_gc_x = int((dem_dim_1 - 1) / pixel_per_gc) - 2 * offset_gc
    if mask is None:
        mask = np.ones((num_gc_y, num_gc_x), dtype=np.uint8)
    if (mask.shape[0] != num_gc_y) or (mask.shape[1] != num_gc_x):
        raise ValueError("shape of mask is inconsistent with other input")
    if mask.dtype != "uint8":
        raise TypeError("data type of mask must be 'uint8'")
    if dist_search < 0.1:
        raise ValueError("'dist_search' must be at least 100.0 m")
    if geom_type not in ("triangle", "quad", "grid"):
        raise ValueError("invalid input argument for geom_type")
    if (ang_max < 85.0) or (ang_max > 89.99):
        raise ValueError("'ang_max' must be in the range [85.0, 89.99]")
    if (sw_dir_cor_max < 2.0) or (sw_dir_cor_max > 100.0):
        raise ValueError("'sw_dir_cor_max' must be in the range [2.0, 100.0]")

    # Check size of input geometries
    if (dem_dim_0 > 32767) or (dem_dim_1 > 32767):
        raise ValueError("maximal allowed input length for dem_dim_0 and "
                         "dem_dim_1 is 32'767")

    # Ensure that passed arrays are contiguous in memory
    vert_grid = np.ascontiguousarray(vert_grid)
    vert_grid_in = np.ascontiguousarray(vert_grid_in)
    sun_pos = np.ascontiguousarray(sun_pos)

    # Convert input strings to bytes
    geom_type_c = geom_type.encode("utf-8")

    # Allocate array for shortwave correction factors
    cdef int len_in_0 = int((dem_dim_in_0 - 1) / pixel_per_gc)
    cdef int len_in_1 = int((dem_dim_in_1 - 1) / pixel_per_gc)
    cdef int dim_sun_0 = sun_pos.shape[0]
    cdef int dim_sun_1 = sun_pos.shape[1]
    cdef np.ndarray[np.float32_t, ndim = 4, mode = "c"] \
        sw_dir_cor = np.empty((len_in_0, len_in_1, dim_sun_0, dim_sun_1),
                              dtype=np.float32)
    sw_dir_cor.fill(0.0)
    # -> ensure that all elements of array 'sw_dir_cor' are 0.0 (crucial
    # because subgrid correction values are iteratively added) -> probably no longer needed with horizon...
    cdef np.ndarray[np.float32_t, ndim = 2, mode = "c"] \
        sky_view_factor = np.empty((len_in_0, len_in_1), dtype=np.float32)

    sw_dir_cor_svf_comp(
        &vert_grid[0],
        dem_dim_0, dem_dim_1,
        &vert_grid_in[0],
        dem_dim_in_0, dem_dim_in_1,
        &sun_pos[0,0,0],
        dim_sun_0, dim_sun_1,
        &sw_dir_cor[0,0,0,0],
        &sky_view_factor[0,0],
        pixel_per_gc,
        offset_gc,
        &mask[0, 0],
        dist_search,
        hori_azim_num,
        geom_type_c,
        ang_max,
        sw_dir_cor_max)

    return sw_dir_cor, sky_view_factor
