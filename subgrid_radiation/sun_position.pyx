# Copyright (c) 2023 ETH Zurich, Christian R. Steger
# MIT License

cimport numpy as np
import numpy as np
import os

cdef extern from "sun_position_comp.h" namespace "shapes":
    cdef cppclass CppTerrain:
        CppTerrain()
        void initialise(float*, int, int, float*, int, int,
                        int, int, unsigned char*, float, char*, float, float)
        void sw_dir_cor(float*, float*)
        void sw_dir_cor_coherent(float*, float*)
        void sw_dir_cor_coherent_rp8(float *, float *)

cdef class Terrain:

    cdef CppTerrain *thisptr

    def __cinit__(self):
        self.thisptr = new CppTerrain()

    def __dealloc__(self):
        del self.thisptr

    def initialise(self,
                   np.ndarray[np.float32_t, ndim = 1] vert_grid,
                   int dem_dim_0, int dem_dim_1,
                   np.ndarray[np.float32_t, ndim = 1] vert_grid_in,
                   int dem_dim_in_0, int dem_dim_in_1,
                   int pixel_per_gc,
                   int offset_gc,
                   np.ndarray[np.uint8_t, ndim = 2] mask,
                   float dist_search=100.0,
                   str geom_type="grid",
                   float sw_dir_cor_max=25.0,
                   float ang_max=89.9):
        """Initialise Terrain class with Digital Elevation Model (DEM) data.

        Parameters
        ----------
        vert_grid : ndarray of float
            Array (one-dimensional) with vertices of DEM in ENU coordinates
            [metre]
        dem_dim_0 : int
            Dimension length of DEM in y-direction
        dem_dim_1 : int
            Dimension length of DEM in x-direction
        vert_grid_in : ndarray of float
            Array (one-dimensional) with vertices of inner DEM with 0.0 m
            elevation in ENU coordinates [metre]
        dem_dim_in_0 : int
            Dimension length of inner DEM in y-direction
        dem_dim_in_1 : int
            Dimension length of inner DEM in x-direction
        pixel_per_gc : int
            Number of subgrid pixels within one grid cell (along one dimension)
        offset_gc : int
            Offset number of grid cells
        mask : ndarray of uint8
            Array (two-dimensional) with grid cells for which 'sw_dir_cor' and
            'sky_view_factor' are computed. Masked (0) grid cells are filled
            with NaN.
        dist_search : float
            Search distance for topographic shadowing [kilometre]
        geom_type : str
            Embree geometry type (triangle, quad, grid)
    sw_dir_cor_max : float
        Maximal allowed correction factor for direct downward shortwave
        radiation [-]
    ang_max : float
        Maximal angle between (I) sun vector and horizontal surface normal
        and (II) sun vector and tilted surface normal for which correction is
        computed. For larger angles, 'sw_dir_cor' is set to 0.0 [degree]"""

        # Check consistency and validity of input arguments
        if ((dem_dim_0 != (2 * offset_gc * pixel_per_gc) + dem_dim_in_0)
                or (dem_dim_1 != (2 * offset_gc * pixel_per_gc)
                    + dem_dim_in_1)):
            raise ValueError("Inconsistency between input arguments "
                             + "'dem_dim_?', 'dem_dim_in_?', 'offset_gc' "
                             + "and 'pixel_per_gc'")
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
        if (sw_dir_cor_max < 2.0) or (sw_dir_cor_max > 100.0):
            raise ValueError(
                "'sw_dir_cor_max' must be in the range [2.0, 100.0]")
        if (ang_max < 88.0) or (ang_max > 89.99):
            raise ValueError("'ang_max' must be in the range [88.0, 89.99]")

        # Check size of input geometries
        if (dem_dim_0 > 32767) or (dem_dim_1 > 32767):
            raise ValueError("maximal allowed input length for dem_dim_0 and "
                             "dem_dim_1 is 32'767")

        self.thisptr.initialise(&vert_grid[0],
                                dem_dim_0, dem_dim_1,
                                &vert_grid_in[0],
                                dem_dim_in_0, dem_dim_in_1,
                                pixel_per_gc,
                                offset_gc,
                                &mask[0, 0],
                                dist_search,
                                geom_type.encode("utf-8"),
                                sw_dir_cor_max,
                                ang_max)

# -----------------------------------------------------------------------------

    def sw_dir_cor(self, np.ndarray[np.float32_t, ndim = 1] sun_pos,
                   np.ndarray[np.float32_t, ndim = 2] sw_dir_cor):
        """Compute subgrid-scale correction factors for direct downward
        shortwave radiation for a specific sun position.

        Parameters
        ----------
        sun_pos : ndarray of float
            Array (one-dimensional) with sun position in ENU coordinates
            (x, y, z) [metre]
        sw_dir_cor : ndarray of float
            Array (two-dimensional) with shortwave correction factor (y, x)
            [-]

        References
        ----------
        - Mueller, M. D., & Scherer, D. (2005): A Grid- and Subgrid-Scale
        Radiation Parameterization of Topographic Effects for Mesoscale
        Weather Forecast Models, Monthly Weather Review, 133(6), 1431-1442."""

        # Check consistency and validity of input arguments
        if (sun_pos.ndim != 1) or (sun_pos.size != 3):
            raise ValueError("array 'sun_pos' has incorrect shape")
        if not sw_dir_cor.flags["C_CONTIGUOUS"]:
            raise ValueError("array 'sw_dir_cor' is not C-contiguous")

        sw_dir_cor.fill(0.0)
        # -> ensure that all elements of array 'sw_dir_cor' are 0.0 (crucial
        # because subgrid correction values are iteratively added)

        self.thisptr.sw_dir_cor(&sun_pos[0], &sw_dir_cor[0,0])

# -----------------------------------------------------------------------------

    def sw_dir_cor_coherent(
            self, np.ndarray[np.float32_t, ndim = 1] sun_pos,
            np.ndarray[np.float32_t, ndim = 2] sw_dir_cor):
        """Compute subgrid-scale correction factors for direct downward
        shortwave radiation for a specific sun position (use coherent rays).

        Parameters
        ----------
        sun_pos : ndarray of float
            Array (one-dimensional) with sun position in ENU coordinates
            (x, y, z) [metre]
        sw_dir_cor : ndarray of float
            Array (two-dimensional) with shortwave correction factor (y, x)
            [-]

        References
        ----------
        - Mueller, M. D., & Scherer, D. (2005): A Grid- and Subgrid-Scale
        Radiation Parameterization of Topographic Effects for Mesoscale
        Weather Forecast Models, Monthly Weather Review, 133(6), 1431-1442."""

        # Check consistency and validity of input arguments
        if (sun_pos.ndim != 1) or (sun_pos.size != 3):
            raise ValueError("array 'sun_pos' has incorrect shape")
        if not sw_dir_cor.flags["C_CONTIGUOUS"]:
            raise ValueError("array 'sw_dir_cor' is not C-contiguous")

        sw_dir_cor.fill(0.0)

        self.thisptr.sw_dir_cor_coherent(&sun_pos[0], &sw_dir_cor[0,0])

# -----------------------------------------------------------------------------

    def sw_dir_cor_coherent_rp8(
            self, np.ndarray[np.float32_t, ndim = 1] sun_pos,
            np.ndarray[np.float32_t, ndim = 2] sw_dir_cor):
        """Compute subgrid-scale correction factors for direct downward
        shortwave radiation for a specific sun position (use coherent rays
        with packages of 8 rays).

        Parameters
        ----------
        sun_pos : ndarray of float
            Array (one-dimensional) with sun position in ENU coordinates
            (x, y, z) [metre]
        sw_dir_cor : ndarray of float
            Array (two-dimensional) with shortwave correction factor (y, x)
            [-]

        References
        ----------
        - Mueller, M. D., & Scherer, D. (2005): A Grid- and Subgrid-Scale
        Radiation Parameterization of Topographic Effects for Mesoscale
        Weather Forecast Models, Monthly Weather Review, 133(6), 1431-1442."""

        # Check consistency and validity of input arguments
        if (sun_pos.ndim != 1) or (sun_pos.size != 3):
            raise ValueError("array 'sun_pos' has incorrect shape")
        if not sw_dir_cor.flags["C_CONTIGUOUS"]:
            raise ValueError("array 'sw_dir_cor' is not C-contiguous")

        sw_dir_cor.fill(0.0)

        self.thisptr.sw_dir_cor_coherent_rp8(&sun_pos[0], &sw_dir_cor[0,0])
