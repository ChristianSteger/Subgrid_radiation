// Copyright (c) 2023 ETH Zurich, Christian R. Steger
// MIT License

#include <cstdio>
#include <embree3/rtcore.h>
#include <stdio.h>
#include <math.h>
#include <limits>
#include <stdio.h>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <string.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <sstream>
#include <iomanip>

using namespace std;

//#############################################################################
// Auxiliary functions
//#############################################################################

// ----------------------------------------------------------------------------
// Unit conversion
// ----------------------------------------------------------------------------

// Convert degree to radian
inline float deg2rad(float ang) {
    /* Parameters
       ----------
       ang: angle [degree]

       Returns
       ----------
       ang: angle [radian]
    */
    return ((ang / 180.0) * M_PI);
}

// Convert radian to degree
inline float rad2deg(float ang) {
    /* Parameters
       ----------
       ang: angle [radian]

       Returns
       ----------
       ang: angle [degree]
    */
    return ((ang / M_PI) * 180.0);
}

// ----------------------------------------------------------------------------
// Compute linear array index from multidimensional subscripts
// ----------------------------------------------------------------------------

// Linear index from subscripts (2D-array)
inline size_t lin_ind_2d(size_t dim_1, size_t ind_0, size_t ind_1) {
    /* Parameters
       ----------
       dim_1: second dimension length of two-dimensional array [-]
       ind_0: first array indices [-]
       ind_1: second array indices [-]

       Returns
       ----------
       ind_lin: linear index of array [-]
    */
    return (ind_0 * dim_1 + ind_1);
}

// Linear index from subscripts (3D-array)
inline size_t lin_ind_3d(size_t dim_1, size_t dim_2,
    size_t ind_0, size_t ind_1, size_t ind_2) {
    /* Parameters
       ----------
       dim_1: second dimension length of three-dimensional array [-]
       dim_2: third dimension length of three-dimensional array [-]
       ind_0: first array indices [-]
       ind_1: second array indices [-]
       ind_2: third array indices [-]

       Returns
       ----------
       ind_lin: linear index of array [-]
    */
    return (ind_0 * (dim_1 * dim_2) + ind_1 * dim_2 + ind_2);
}

// Linear index from subscripts (4D-array)
inline size_t lin_ind_4d(size_t dim_1, size_t dim_2, size_t dim_3,
    size_t ind_0, size_t ind_1, size_t ind_2, size_t ind_3) {
    /* Parameters
       ----------
       dim_1: second dimension length of four-dimensional array [-]
       dim_2: third dimension length of four-dimensional array [-]
       dim_3: fourth dimension length of four-dimensional array [-]
       ind_0: first array indices [-]
       ind_1: second array indices [-]
       ind_2: third array indices [-]
       ind_3: fourth array indicies [-]

       Returns
       ----------
       ind_lin: linear index of array [-]
    */
    return (ind_0 * (dim_1 * dim_2 * dim_3) + ind_1 * (dim_2 * dim_3)
        + ind_2 * dim_3 + ind_3);
}

// ----------------------------------------------------------------------------
// Vector and matrix operations
// ----------------------------------------------------------------------------

// Unit vector
inline void vec_unit(float &v_x, float &v_y, float &v_z) {
    /* Parameters
       ----------
       v_x: x-component of vector [arbitrary]
       v_y: y-component of vector [arbitrary]
       v_z: z-component of vector [arbitrary]
    */
    float mag = sqrt(v_x * v_x + v_y * v_y + v_z * v_z);
    v_x = v_x / mag;
    v_y = v_y / mag;
    v_z = v_z / mag;
}

// ----------------------------------------------------------------------------
// Triangle operations
// ----------------------------------------------------------------------------

// Triangle surface normal and area
inline void triangle_normal_area(
    float &vert_0_x, float &vert_0_y, float &vert_0_z,
    float &vert_1_x, float &vert_1_y, float &vert_1_z,
    float &vert_2_x, float &vert_2_y, float &vert_2_z,
    float &norm_x, float &norm_y, float &norm_z,
    float &area) {
    /* Parameters
       ----------
       vert_0_x: x-component of first triangle vertices [m]
       vert_0_y: y-component of first triangle vertices [m]
       vert_0_z: z-component of first triangle vertices [m]
       vert_1_x: x-component of second triangle vertices [m]
       vert_1_y: y-component of second triangle vertices [m]
       vert_1_z: z-component of second triangle vertices [m]
       vert_2_x: x-component of third triangle vertices [m]
       vert_2_y: y-component of third triangle vertices [m]
       vert_2_z: z-component of third triangle vertices [m]
       norm_x: x-component of triangle surface normal [-]
       norm_y: y-component of triangle surface normal [-]
       norm_z: z-component of triangle surface normal [-]
       area: area of triangle [m2]
    */
    float a_x = vert_2_x - vert_1_x;
    float a_y = vert_2_y - vert_1_y;
    float a_z = vert_2_z - vert_1_z;
    float b_x = vert_0_x - vert_1_x;
    float b_y = vert_0_y - vert_1_y;
    float b_z = vert_0_z - vert_1_z;

    norm_x = a_y * b_z - a_z * b_y;
    norm_y = a_z * b_x - a_x * b_z;
    norm_z = a_x * b_y - a_y * b_x;

    float mag = sqrt(norm_x * norm_x + norm_y * norm_y + norm_z * norm_z);
    norm_x = norm_x / mag;
    norm_y = norm_y / mag;
    norm_z = norm_z / mag;

    area = mag / 2.0;
}

// Triangle centroid
inline void triangle_centroid(
    float &vert_0_x, float &vert_0_y, float &vert_0_z,
    float &vert_1_x, float &vert_1_y, float &vert_1_z,
    float &vert_2_x, float &vert_2_y, float &vert_2_z,
    float &cent_x, float &cent_y, float &cent_z) {
    /* Parameters
       ----------
       vert_0_x: x-component of first triangle vertices [m]
       vert_0_y: y-component of first triangle vertices [m]
       vert_0_z: z-component of first triangle vertices [m]
       vert_1_x: x-component of second triangle vertices [m]
       vert_1_y: y-component of second triangle vertices [m]
       vert_1_z: z-component of second triangle vertices [m]
       vert_2_x: x-component of third triangle vertices [m]
       vert_2_y: y-component of third triangle vertices [m]
       vert_2_z: z-component of third triangle vertices [m]
       cent_x: x-component of triangle centroid [-]
       cent_y: y-component of triangle centroid [-]
        cent_z: z-component of triangle centroid [-]
    */
    cent_x = (vert_0_x + vert_1_x + vert_2_x) / 3.0;
    cent_y = (vert_0_y + vert_1_y + vert_2_y) / 3.0;
    cent_z = (vert_0_z + vert_1_z + vert_2_z) / 3.0;
}

// Vertices of lower left triangle (within pixel)
inline void triangle_vert_ll(size_t dim_1, size_t ind_0, size_t ind_1,
    size_t &ind_tri_0, size_t &ind_tri_1, size_t &ind_tri_2) {
    /* Parameters
       ----------

    */
    ind_tri_0 = (ind_0 * dim_1 + ind_1) * 3;
    ind_tri_1 = (ind_0 * dim_1 + ind_1 + 1) * 3;
    ind_tri_2 = ((ind_0 + 1) * dim_1 + ind_1) * 3;
}

// Vertices of upper right triangle (within pixel)
inline void triangle_vert_ur(size_t dim_1, size_t ind_0, size_t ind_1,
    size_t &ind_tri_0, size_t &ind_tri_1, size_t &ind_tri_2) {
    /* Parameters
       ----------

    */
    ind_tri_0 = (ind_0 * dim_1 + ind_1 + 1) * 3;
    ind_tri_1 = ((ind_0 + 1) * dim_1 + ind_1 + 1) * 3;
    ind_tri_2 = ((ind_0 + 1) * dim_1 + ind_1) * 3;

}

// Store above two functions in array
void (*func_ptr[2])(size_t dim_1, size_t ind_0, size_t ind_1,
    size_t &ind_tri_0, size_t &ind_tri_1, size_t &ind_tri_2)
    = {triangle_vert_ll, triangle_vert_ur};

//#############################################################################
// Miscellaneous
//#############################################################################

// Namespace
#if defined(RTC_NAMESPACE_USE)
    RTC_NAMESPACE_USE
#endif

// Error function
void errorFunction(void* userPtr, enum RTCError error, const char* str) {
    printf("error %d: %s\n", error, str);
}

// Initialisation of device and registration of error handler
RTCDevice initializeDevice() {
    RTCDevice device = rtcNewDevice(NULL);
    if (!device) {
        printf("error %d: cannot create device\n", rtcGetDeviceError(NULL));
    }
    rtcSetDeviceErrorFunction(device, errorFunction, NULL);
    return device;
}

//#############################################################################
// Create scene from geometries
//#############################################################################

// Structures for triangle and quad
struct Triangle { int v0, v1, v2; };
struct Quad { int v0, v1, v2, v3; };
// -> above structures must contain 32-bit integers (-> Embree documentation).
//    Theoretically, these integers should be unsigned but the binary
//    representation until 2'147'483'647 is identical between signed/unsigned
//    integer.

// Initialise scene
RTCScene initializeScene(RTCDevice device, float* vert_grid,
    int dem_dim_0, int dem_dim_1, char* geom_type) {

    RTCScene scene = rtcNewScene(device);
    rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);

    int num_vert = (dem_dim_0 * dem_dim_1);
    printf("DEM dimensions: (%d, %d) \n", dem_dim_0, dem_dim_1);
    printf("Number of vertices: %d \n", num_vert);

    RTCGeometryType rtc_geom_type;
    if (strcmp(geom_type, "triangle") == 0) {
        rtc_geom_type = RTC_GEOMETRY_TYPE_TRIANGLE;
    } else if (strcmp(geom_type, "quad") == 0) {
        rtc_geom_type = RTC_GEOMETRY_TYPE_QUAD;
    } else {
        rtc_geom_type = RTC_GEOMETRY_TYPE_GRID;
    }

    RTCGeometry geom = rtcNewGeometry(device, rtc_geom_type);
    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
        RTC_FORMAT_FLOAT3, vert_grid, 0, 3*sizeof(float), num_vert);

    //-------------------------------------------------------------------------
    // Triangle
    //-------------------------------------------------------------------------
    if (strcmp(geom_type, "triangle") == 0) {
        cout << "Selected geometry type: triangle" << endl;
        int num_tri = ((dem_dim_0 - 1) * (dem_dim_1 - 1)) * 2;
        printf("Number of triangles: %d \n", num_tri);
        Triangle* triangles = (Triangle*) rtcSetNewGeometryBuffer(geom,
            RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Triangle),
            num_tri);
        int n = 0;
        for (int i = 0; i < (dem_dim_0 - 1); i++) {
            for (int j = 0; j < (dem_dim_1 - 1); j++) {
                triangles[n].v0 = (i * dem_dim_1) + j;
                triangles[n].v1 = (i * dem_dim_1) + j + 1;
                triangles[n].v2 = ((i + 1) * dem_dim_1) + j;
                n++;
                triangles[n].v0 = (i * dem_dim_1) + j + 1;
                triangles[n].v1 = ((i + 1) * dem_dim_1) + j + 1;
                triangles[n].v2 = ((i + 1) * dem_dim_1) + j;
                n++;
            }
        }
    //-------------------------------------------------------------------------
    // Quad
    //-------------------------------------------------------------------------
    } else if (strcmp(geom_type, "quad") == 0) {
        cout << "Selected geometry type: quad" << endl;
        int num_quad = ((dem_dim_0 - 1) * (dem_dim_1 - 1));
        printf("Number of quads: %d \n", num_quad);
        Quad* quads = (Quad*) rtcSetNewGeometryBuffer(geom,
            RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT4, sizeof(Quad),
            num_quad);
        int n = 0;
        for (int i = 0; i < (dem_dim_0 - 1); i++) {
            for (int j = 0; j < (dem_dim_1 - 1); j++) {
                //  identical to grid scene (-> otherwise reverse v0, v1, ...)
                quads[n].v0 = (i * dem_dim_1) + j;
                quads[n].v1 = (i * dem_dim_1) + j + 1;
                quads[n].v2 = ((i + 1) * dem_dim_1) + j + 1;
                quads[n].v3 = ((i + 1) * dem_dim_1) + j;
                n++;
            }
        }
    //-------------------------------------------------------------------------
    // Grid
    //-------------------------------------------------------------------------
    } else {
        cout << "Selected geometry type: grid" << endl;
        RTCGrid* grid = (RTCGrid*)rtcSetNewGeometryBuffer(geom,
            RTC_BUFFER_TYPE_GRID, 0, RTC_FORMAT_GRID, sizeof(RTCGrid), 1);
        grid[0].startVertexID = 0;
        grid[0].stride        = dem_dim_1;
        grid[0].width         = dem_dim_1;
        grid[0].height        = dem_dim_0;
    }
    //-------------------------------------------------------------------------

    auto start = std::chrono::high_resolution_clock::now();

    // Commit geometry
    rtcCommitGeometry(geom);

    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);

    //-------------------------------------------------------------------------

    // Commit scene
    rtcCommitScene(scene);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = end - start;
    cout << "BVH build time: " << time.count() << " s" << endl;

    return scene;

}

//#############################################################################
// Main functions
//#############################################################################

//-----------------------------------------------------------------------------
// Default
//-----------------------------------------------------------------------------

void sw_dir_cor_comp(
    float* vert_grid,
    int dem_dim_0, int dem_dim_1,
    float* vert_grid_in,
    int dem_dim_in_0, int dem_dim_in_1,
    float* sun_pos,
    int dim_sun_0, int dim_sun_1,
    float* sw_dir_cor,
    int pixel_per_gc,
    int offset_gc,
    uint8_t* mask,
    float dist_search,
    char* geom_type,
    float sw_dir_cor_max,
    float ang_max) {

    cout << "--------------------------------------------------------" << endl;
    cout << "Compute lookup table with default method" << endl;
    cout << "--------------------------------------------------------" << endl;

    // Hard-coded settings
    float ray_org_elev = 0.05;
    // value to elevate ray origin (-> avoids potential issue with numerical
    // imprecision / truncation) [m]

    // Number of grid cells
    int num_gc_y = (dem_dim_in_0 - 1) / pixel_per_gc;
    int num_gc_x = (dem_dim_in_1 - 1) / pixel_per_gc;
    cout << "Number of grid cells in y-direction: " << num_gc_y
        << endl;
    cout << "Number of grid cells in x-direction: " << num_gc_x << endl;

    // Number of triangles
    int num_tri = (dem_dim_in_0 - 1) * (dem_dim_in_1 - 1) * 2;
    cout << "Number of triangles: " << num_tri << endl;

    // Unit conversion(s)
    float dot_prod_min = cos(deg2rad(ang_max));
    dist_search *= 1000.0;  // [kilometre] to [metre]
    cout << "Search distance: " << dist_search << " m" << endl;

    cout << "ang_max: " << ang_max << " degree" << endl;
    cout << "sw_dir_cor_max: " << sw_dir_cor_max  << endl;

    // Initialisation
    auto start_ini = std::chrono::high_resolution_clock::now();
    RTCDevice device = initializeDevice();
    RTCScene scene = initializeScene(device, vert_grid, dem_dim_0, dem_dim_1,
        geom_type);
    auto end_ini = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = end_ini - start_ini;
    cout << "Total initialisation time: " << time.count() << " s" << endl;

    //-------------------------------------------------------------------------

    auto start_ray = std::chrono::high_resolution_clock::now();
    size_t num_rays = 0;

    num_rays += tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, num_gc_y), 0.0,
        [&](tbb::blocked_range<size_t> r, size_t num_rays) {  // parallel

    // Loop through 2D-field of grid cells
    //for (size_t i = 0; i < num_gc_y; i++) {  // serial
    for (size_t i=r.begin(); i<r.end(); ++i) {  // parallel
        for (size_t j = 0; j < num_gc_x; j++) {

            size_t lin_ind_gc = lin_ind_2d(num_gc_x, i, j);
            if (mask[lin_ind_gc] == 1) {

            // Loop through 2D-field of DEM pixels
            for (size_t k = (i * pixel_per_gc);
                k < ((i * pixel_per_gc) + pixel_per_gc); k++) {
                for (size_t m = (j * pixel_per_gc);
                    m < ((j * pixel_per_gc) + pixel_per_gc); m++) {

                    // Loop through two triangles per pixel
                    for (size_t n = 0; n < 2; n++) {

                        //-----------------------------------------------------
                        // Tilted triangle
                        //-----------------------------------------------------

                        size_t ind_tri_0, ind_tri_1, ind_tri_2;
                        func_ptr[n](dem_dim_1,
                            k + (pixel_per_gc * offset_gc),
                            m + (pixel_per_gc * offset_gc),
                            ind_tri_0, ind_tri_1, ind_tri_2);

                        float vert_0_x = vert_grid[ind_tri_0];
                        float vert_0_y = vert_grid[ind_tri_0 + 1];
                        float vert_0_z = vert_grid[ind_tri_0 + 2];
                        float vert_1_x = vert_grid[ind_tri_1];
                        float vert_1_y = vert_grid[ind_tri_1 + 1];
                        float vert_1_z = vert_grid[ind_tri_1 + 2];
                        float vert_2_x = vert_grid[ind_tri_2];
                        float vert_2_y = vert_grid[ind_tri_2 + 1];
                        float vert_2_z = vert_grid[ind_tri_2 + 2];

                        float cent_x, cent_y, cent_z;
                        triangle_centroid(vert_0_x, vert_0_y, vert_0_z,
                            vert_1_x, vert_1_y, vert_1_z,
                            vert_2_x, vert_2_y, vert_2_z,
                            cent_x, cent_y, cent_z);

                        float norm_tilt_x, norm_tilt_y, norm_tilt_z, area_tilt;
                        triangle_normal_area(vert_0_x, vert_0_y, vert_0_z,
                            vert_1_x, vert_1_y, vert_1_z,
                            vert_2_x, vert_2_y, vert_2_z,
                            norm_tilt_x, norm_tilt_y, norm_tilt_z,
                            area_tilt);

                        // Ray origin
                        float ray_org_x = (cent_x
                            + norm_tilt_x * ray_org_elev);
                        float ray_org_y = (cent_y
                            + norm_tilt_y * ray_org_elev);
                        float ray_org_z = (cent_z
                            + norm_tilt_z * ray_org_elev);

                        //-----------------------------------------------------
                        // Horizontal triangle
                        //-----------------------------------------------------

                        func_ptr[n](dem_dim_in_1, k, m,
                            ind_tri_0, ind_tri_1, ind_tri_2);

                        vert_0_x = vert_grid_in[ind_tri_0];
                        vert_0_y = vert_grid_in[ind_tri_0 + 1];
                        vert_0_z = vert_grid_in[ind_tri_0 + 2];
                        vert_1_x = vert_grid_in[ind_tri_1];
                        vert_1_y = vert_grid_in[ind_tri_1 + 1];
                        vert_1_z = vert_grid_in[ind_tri_1 + 2];
                        vert_2_x = vert_grid_in[ind_tri_2];
                        vert_2_y = vert_grid_in[ind_tri_2 + 1];
                        vert_2_z = vert_grid_in[ind_tri_2 + 2];

                        float norm_hori_x, norm_hori_y, norm_hori_z, area_hori;
                        triangle_normal_area(vert_0_x, vert_0_y, vert_0_z,
                            vert_1_x, vert_1_y, vert_1_z,
                            vert_2_x, vert_2_y, vert_2_z,
                            norm_hori_x, norm_hori_y, norm_hori_z,
                            area_hori);

                        float surf_enl_fac = area_tilt / area_hori;

                        //-----------------------------------------------------
                        // Loop through sun positions and compute correction
                        // factors
                        //-----------------------------------------------------

                        size_t ind_lin_sun, ind_lin_cor;
                        for (size_t o = 0; o < dim_sun_0; o++) {
                            for (size_t p = 0; p < dim_sun_1; p++) {

                                ind_lin_sun = lin_ind_3d(dim_sun_1, 3,
                                    o, p, 0);
                                ind_lin_cor = lin_ind_4d(num_gc_x,
                                    dim_sun_0, dim_sun_1, i, j, o, p);

                                // Compute sun unit vector
                                float sun_x = (sun_pos[ind_lin_sun]
                                    - ray_org_x);
                                float sun_y = (sun_pos[ind_lin_sun + 1]
                                    - ray_org_y);
                                float sun_z = (sun_pos[ind_lin_sun + 2]
                                    - ray_org_z);
                                vec_unit(sun_x, sun_y, sun_z);

                                // Check for self-shadowing (Earth)
                                float dot_prod_hs = (norm_hori_x * sun_x
                                    + norm_hori_y * sun_y
                                    + norm_hori_z * sun_z);
                                if (dot_prod_hs <= dot_prod_min) {
                                    continue;  // sw_dir_cor += 0.0
                                }

                                // Check for self-shadowing (triangle)
                                float dot_prod_ts = norm_tilt_x * sun_x
                                    + norm_tilt_y * sun_y
                                    + norm_tilt_z * sun_z;
                                if (dot_prod_ts <= 0.0) {
                                    continue;  // sw_dir_cor += 0.0
                                }

                                // Intersect context
                                struct RTCIntersectContext context;
                                rtcInitIntersectContext(&context);

                                // Ray structure
                                struct RTCRay ray;
                                ray.org_x = ray_org_x;
                                ray.org_y = ray_org_y;
                                ray.org_z = ray_org_z;
                                ray.dir_x = sun_x;
                                ray.dir_y = sun_y;
                                ray.dir_z = sun_z;
                                ray.tnear = 0.0;
                                ray.tfar = dist_search;
                                // std::numeric_limits<float>::infinity();

                                // Intersect ray with scene
                                rtcOccluded1(scene, &context, &ray);
                                if (ray.tfar > 0.0) {
                                    // no intersection -> 'tfar' is not
                                    // updated; otherwise 'tfar' = -inf
                                    sw_dir_cor[ind_lin_cor] =
                                        sw_dir_cor[ind_lin_cor]
                                        + std::min(((dot_prod_ts
                                        / dot_prod_hs) * surf_enl_fac),
                                        sw_dir_cor_max);
                                }  // else: sw_dir_cor += 0.0
                                num_rays += 1;

                            }
                        }

                    }

                }
            }

            } else {

                size_t ind_lin = lin_ind_4d(num_gc_x, dim_sun_0, dim_sun_1,
                    i, j, 0, 0);
                for (size_t k = 0; k < (dim_sun_0 * dim_sun_1) ; k++) {
                    sw_dir_cor[ind_lin + k] = NAN;
                }

            }

        }
    }

    return num_rays;  // parallel
    }, std::plus<size_t>());  // parallel

    auto end_ray = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_ray = (end_ray - start_ray);
    cout << "Ray tracing time: " << time_ray.count() << " s" << endl;
    cout << "Number of rays shot: " << num_rays << endl;
    float frac_ray = (float)num_rays /
        ((float)num_tri * (float)dim_sun_0 * (float)dim_sun_1);
    cout << "Fraction of rays required: " << frac_ray << endl;

    // Divide accumulated values by number of triangles within grid cell
    float num_tri_per_gc = pixel_per_gc * pixel_per_gc * 2.0;
    size_t num_elem = (num_gc_y * num_gc_x * dim_sun_0 * dim_sun_1);
    for (size_t i = 0; i < num_elem; i++) {
        sw_dir_cor[i] /= num_tri_per_gc;
    }

    // Release resources allocated through Embree
    rtcReleaseScene(scene);
    rtcReleaseDevice(device);

    auto end_tot = std::chrono::high_resolution_clock::now();
    time = end_tot - start_ini;
    cout << "Total run time: " << time.count() << " s" << endl;

    //-------------------------------------------------------------------------

    cout << "--------------------------------------------------------" << endl;

}

//-----------------------------------------------------------------------------
// Use coherent rays
//-----------------------------------------------------------------------------

void sw_dir_cor_comp_coherent(
    float* vert_grid,
    int dem_dim_0, int dem_dim_1,
    float* vert_grid_in,
    int dem_dim_in_0, int dem_dim_in_1,
    float* sun_pos,
    int dim_sun_0, int dim_sun_1,
    float* sw_dir_cor,
    int pixel_per_gc,
    int offset_gc,
    uint8_t* mask,
    float dist_search,
    char* geom_type,
    float sw_dir_cor_max,
    float ang_max) {

    cout << "--------------------------------------------------------" << endl;
    cout << "Compute lookup table with coherent rays" << endl;
    cout << "--------------------------------------------------------" << endl;

    // Hard-coded settings
    float ray_org_elev = 0.05;
    // value to elevate ray origin (-> avoids potential issue with numerical
    // imprecision / truncation) [m]

    // Number of grid cells
    int num_gc_y = (dem_dim_in_0 - 1) / pixel_per_gc;
    int num_gc_x = (dem_dim_in_1 - 1) / pixel_per_gc;
    cout << "Number of grid cells in y-direction: " << num_gc_y
        << endl;
    cout << "Number of grid cells in x-direction: " << num_gc_x << endl;

    // Number of triangles
    int num_tri = (dem_dim_in_0 - 1) * (dem_dim_in_1 - 1) * 2;
    cout << "Number of triangles: " << num_tri << endl;
    int num_tri_per_gc = pixel_per_gc * pixel_per_gc * 2;

    // Unit conversion(s)
    float dot_prod_min = cos(deg2rad(ang_max));
    dist_search *= 1000.0;  // [kilometre] to [metre]
    cout << "Search distance: " << dist_search << " m" << endl;

    cout << "ang_max: " << ang_max << " degree" << endl;
    cout << "sw_dir_cor_max: " << sw_dir_cor_max  << endl;

    // Initialisation
    auto start_ini = std::chrono::high_resolution_clock::now();
    RTCDevice device = initializeDevice();
    RTCScene scene = initializeScene(device, vert_grid, dem_dim_0, dem_dim_1,
        geom_type);
    auto end_ini = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = end_ini - start_ini;
    cout << "Total initialisation time: " << time.count() << " s" << endl;

    //-------------------------------------------------------------------------

    auto start_ray = std::chrono::high_resolution_clock::now();
    size_t num_rays = 0;

    num_rays += tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, num_gc_y), 0.0,
        [&](tbb::blocked_range<size_t> r, size_t num_rays) {  // parallel

    // Loop through 2D-field of grid cells
    //for (size_t i = 0; i < num_gc_y; i++) {  // serial
    for (size_t i=r.begin(); i<r.end(); ++i) {  // parallel
        for (size_t j = 0; j < num_gc_x; j++) {

            size_t lin_ind_gc = lin_ind_2d(num_gc_x, i, j);
            if (mask[lin_ind_gc] == 1) {

            float* norm_tilt = new float[num_tri_per_gc * 3];
            float* ray_org = new float[num_tri_per_gc * 3];
            float* norm_hori = new float[num_tri_per_gc * 3];
            float* surf_enl_fac = new float[num_tri_per_gc];
            float* sw_dir_cor_ray = new float[num_tri_per_gc];

            // Compute triangle's centroid, surface normal and area
            size_t ind_incr_3 = 0;
            size_t ind_incr_1 = 0;
            for (size_t k = (i * pixel_per_gc);
                k < ((i * pixel_per_gc) + pixel_per_gc); k++) {
                for (size_t m = (j * pixel_per_gc);
                    m < ((j * pixel_per_gc) + pixel_per_gc); m++) {

                    // Loop through two triangles per pixel
                    for (size_t n = 0; n < 2; n++) {

                        //-----------------------------------------------------
                        // Tilted triangle
                        //-----------------------------------------------------

                        size_t ind_tri_0, ind_tri_1, ind_tri_2;
                        func_ptr[n](dem_dim_1,
                            k + (pixel_per_gc * offset_gc),
                            m + (pixel_per_gc * offset_gc),
                            ind_tri_0, ind_tri_1, ind_tri_2);

                        float vert_0_x = vert_grid[ind_tri_0];
                        float vert_0_y = vert_grid[ind_tri_0 + 1];
                        float vert_0_z = vert_grid[ind_tri_0 + 2];
                        float vert_1_x = vert_grid[ind_tri_1];
                        float vert_1_y = vert_grid[ind_tri_1 + 1];
                        float vert_1_z = vert_grid[ind_tri_1 + 2];
                        float vert_2_x = vert_grid[ind_tri_2];
                        float vert_2_y = vert_grid[ind_tri_2 + 1];
                        float vert_2_z = vert_grid[ind_tri_2 + 2];

                        float cent_x, cent_y, cent_z;
                        triangle_centroid(vert_0_x, vert_0_y, vert_0_z,
                            vert_1_x, vert_1_y, vert_1_z,
                            vert_2_x, vert_2_y, vert_2_z,
                            cent_x, cent_y, cent_z);

                        float norm_tilt_x, norm_tilt_y, norm_tilt_z, area_tilt;
                        triangle_normal_area(vert_0_x, vert_0_y, vert_0_z,
                            vert_1_x, vert_1_y, vert_1_z,
                            vert_2_x, vert_2_y, vert_2_z,
                            norm_tilt_x, norm_tilt_y, norm_tilt_z,
                            area_tilt);
                        norm_tilt[ind_incr_3] = norm_tilt_x;
                        norm_tilt[ind_incr_3 + 1] = norm_tilt_y;
                        norm_tilt[ind_incr_3 + 2] = norm_tilt_z;

                        // Ray origin
                        ray_org[ind_incr_3] = (cent_x
                            + norm_tilt_x * ray_org_elev);
                        ray_org[ind_incr_3 + 1] = (cent_y
                            + norm_tilt_y * ray_org_elev);
                        ray_org[ind_incr_3 + 2] = (cent_z
                            + norm_tilt_z * ray_org_elev);

                        //-----------------------------------------------------
                        // Horizontal triangle
                        //-----------------------------------------------------

                        func_ptr[n](dem_dim_in_1, k, m,
                            ind_tri_0, ind_tri_1, ind_tri_2);

                        vert_0_x = vert_grid_in[ind_tri_0];
                        vert_0_y = vert_grid_in[ind_tri_0 + 1];
                        vert_0_z = vert_grid_in[ind_tri_0 + 2];
                        vert_1_x = vert_grid_in[ind_tri_1];
                        vert_1_y = vert_grid_in[ind_tri_1 + 1];
                        vert_1_z = vert_grid_in[ind_tri_1 + 2];
                        vert_2_x = vert_grid_in[ind_tri_2];
                        vert_2_y = vert_grid_in[ind_tri_2 + 1];
                        vert_2_z = vert_grid_in[ind_tri_2 + 2];

                        float norm_hori_x, norm_hori_y, norm_hori_z, area_hori;
                        triangle_normal_area(vert_0_x, vert_0_y, vert_0_z,
                            vert_1_x, vert_1_y, vert_1_z,
                            vert_2_x, vert_2_y, vert_2_z,
                            norm_hori_x, norm_hori_y, norm_hori_z,
                            area_hori);
                        norm_hori[ind_incr_3] = norm_hori_x;
                        norm_hori[ind_incr_3 + 1] = norm_hori_y;
                        norm_hori[ind_incr_3 + 2] = norm_hori_z;

                        surf_enl_fac[ind_incr_1] = area_tilt / area_hori;

                        ind_incr_3 = ind_incr_3 + 3;
                        ind_incr_1 = ind_incr_1 + 1;

                    }

                }
            }

            //-----------------------------------------------------------------
            // Loop through sun positions and compute correction factors
            //-----------------------------------------------------------------

            RTCRay rays[num_tri_per_gc];

            size_t ind_lin_sun;
            for (size_t o = 0; o < dim_sun_0; o++) {
                for (size_t p = 0; p < dim_sun_1; p++) {

                    ind_lin_sun = lin_ind_3d(dim_sun_1, 3, o, p, 0);

                    //---------------------------------------------------------
                    // Compute correction factors (I)
                    //---------------------------------------------------------

                    ind_incr_3 = 0;
                    ind_incr_1 = 0;
                    unsigned int num_rays_gc = 0;
                    for (size_t k = (i * pixel_per_gc);
                        k < ((i * pixel_per_gc) + pixel_per_gc); k++) {
                        for (size_t m = (j * pixel_per_gc);
                            m < ((j * pixel_per_gc) + pixel_per_gc); m++) {

                            // Loop through two triangles per pixel
                            for (size_t n = 0; n < 2; n++) {

                                // Compute sun unit vector
                                float sun_x = (sun_pos[ind_lin_sun]
                                    - ray_org[ind_incr_3]);
                                float sun_y = (sun_pos[ind_lin_sun + 1]
                                    - ray_org[ind_incr_3 + 1]);
                                float sun_z = (sun_pos[ind_lin_sun + 2]
                                    - ray_org[ind_incr_3 + 2]);
                                vec_unit(sun_x, sun_y, sun_z);

                                // Check for self-shadowing (Earth)
                                float dot_prod_hs
                                    = (norm_hori[ind_incr_3] * sun_x
                                    + norm_hori[ind_incr_3 + 1] * sun_y
                                    + norm_hori[ind_incr_3 + 2] * sun_z);
                                if (dot_prod_hs <= dot_prod_min) {
                                    ind_incr_3 = ind_incr_3 + 3;
                                    ind_incr_1 = ind_incr_1 + 1;
                                    continue;  // sw_dir_cor += 0.0
                                }

                                // Check for self-shadowing (triangle)
                                float dot_prod_ts
                                    = norm_tilt[ind_incr_3] * sun_x
                                    + norm_tilt[ind_incr_3 + 1] * sun_y
                                    + norm_tilt[ind_incr_3 + 2] * sun_z;
                                if (dot_prod_ts <= 0.0) {
                                    ind_incr_3 = ind_incr_3 + 3;
                                    ind_incr_1 = ind_incr_1 + 1;
                                    continue;  // sw_dir_cor += 0.0
                                }

                                // Add ray
                                rays[num_rays_gc].org_x
                                    = ray_org[ind_incr_3];
                                rays[num_rays_gc].org_y
                                    = ray_org[ind_incr_3 + 1];
                                rays[num_rays_gc].org_z
                                    = ray_org[ind_incr_3 + 2];
                                rays[num_rays_gc].dir_x = sun_x;
                                rays[num_rays_gc].dir_y = sun_y;
                                rays[num_rays_gc].dir_z = sun_z;
                                rays[num_rays_gc].tnear = 0.0;
                                rays[num_rays_gc].tfar = dist_search;
                                // std::numeric_limits<float>::infinity();
                                rays[num_rays_gc].id = num_rays_gc;

                                sw_dir_cor_ray[num_rays_gc] =
                                    std::min(((dot_prod_ts / dot_prod_hs)
                                    * surf_enl_fac[ind_incr_1]),
                                    sw_dir_cor_max);
                                num_rays_gc = num_rays_gc + 1;

                                ind_incr_3 = ind_incr_3 + 3;
                                ind_incr_1 = ind_incr_1 + 1;

                            }

                        }
                    }

                    //---------------------------------------------------------
                    // Compute correction factors (II)
                    //---------------------------------------------------------

                    struct RTCIntersectContext context;
                    rtcInitIntersectContext(&context);
                    context.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;

                    // Intersect rays with scene
                    rtcOccluded1M(scene, &context,(RTCRay*)rays, num_rays_gc,
                        sizeof(RTCRay));
                    num_rays += num_rays_gc;

                    float sw_dir_cor_agg = 0.0;
                    for (size_t k = 0; k < num_rays_gc; k++) {
                        if (rays[k].tfar > 0.0) {
                            // no intersection -> 'tfar' is not updated;
                            // otherwise 'tfar' = -inf
                            sw_dir_cor_agg = sw_dir_cor_agg
                                + sw_dir_cor_ray[rays[k].id];
                        }  // else: sw_dir_cor += 0.0
                    }

                    size_t ind_lin_cor = lin_ind_4d(num_gc_x,
                        dim_sun_0, dim_sun_1, i, j, o, p);
                    sw_dir_cor[ind_lin_cor] = sw_dir_cor_agg / num_tri_per_gc;

                }
            }

            delete[] norm_tilt;
            delete[] ray_org;
            delete[] norm_hori;
            delete[] surf_enl_fac;
            delete[] sw_dir_cor_ray;

            } else {

                size_t ind_lin = lin_ind_4d(num_gc_x, dim_sun_0, dim_sun_1,
                    i, j, 0, 0);
                for (size_t k = 0; k < (dim_sun_0 * dim_sun_1) ; k++) {
                    sw_dir_cor[ind_lin + k] = NAN;
                }

            }

        }
    }

    return num_rays;  // parallel
    }, std::plus<size_t>());  // parallel

    auto end_ray = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_ray = (end_ray - start_ray);
    cout << "Ray tracing time: " << time_ray.count() << " s" << endl;
    cout << "Number of rays shot: " << num_rays << endl;
    float frac_ray = (float)num_rays /
        ((float)num_tri * (float)dim_sun_0 * (float)dim_sun_1);
    cout << "Fraction of rays required: " << frac_ray << endl;

    // Release resources allocated through Embree
    rtcReleaseScene(scene);
    rtcReleaseDevice(device);

    auto end_tot = std::chrono::high_resolution_clock::now();
    time = end_tot - start_ini;
    cout << "Total run time: " << time.count() << " s" << endl;

    //-------------------------------------------------------------------------

    cout << "--------------------------------------------------------" << endl;

}

//-----------------------------------------------------------------------------
// Use coherent rays (packages with 8 rays)
//-----------------------------------------------------------------------------

void sw_dir_cor_comp_coherent_rp8(
    float* vert_grid,
    int dem_dim_0, int dem_dim_1,
    float* vert_grid_in,
    int dem_dim_in_0, int dem_dim_in_1,
    float* sun_pos,
    int dim_sun_0, int dim_sun_1,
    float* sw_dir_cor,
    int pixel_per_gc,
    int offset_gc,
    uint8_t* mask,
    float dist_search,
    char* geom_type,
    float sw_dir_cor_max,
    float ang_max) {

    cout << "--------------------------------------------------------" << endl;
    cout << "Compute lookup table with coherent rays" << endl;
    cout << "(packages with 8 rays)" << endl;
    cout << "--------------------------------------------------------" << endl;

    // Hard-coded settings
    float ray_org_elev = 0.05;
    // value to elevate ray origin (-> avoids potential issue with numerical
    // imprecision / truncation) [m]

    // Number of grid cells
    int num_gc_y = (dem_dim_in_0 - 1) / pixel_per_gc;
    int num_gc_x = (dem_dim_in_1 - 1) / pixel_per_gc;
    cout << "Number of grid cells in y-direction: " << num_gc_y
        << endl;
    cout << "Number of grid cells in x-direction: " << num_gc_x << endl;

    // Number of triangles
    int num_tri = (dem_dim_in_0 - 1) * (dem_dim_in_1 - 1) * 2;
    cout << "Number of triangles: " << num_tri << endl;

    // Unit conversion(s)
    float dot_prod_min = cos(deg2rad(ang_max));
    dist_search *= 1000.0;  // [kilometre] to [metre]
    cout << "Search distance: " << dist_search << " m" << endl;

    cout << "ang_max: " << ang_max << " degree" << endl;
    cout << "sw_dir_cor_max: " << sw_dir_cor_max  << endl;

    // Initialisation
    auto start_ini = std::chrono::high_resolution_clock::now();
    RTCDevice device = initializeDevice();
    RTCScene scene = initializeScene(device, vert_grid, dem_dim_0, dem_dim_1,
        geom_type);
    auto end_ini = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = end_ini - start_ini;
    cout << "Total initialisation time: " << time.count() << " s" << endl;

    //-------------------------------------------------------------------------

    auto start_ray = std::chrono::high_resolution_clock::now();
    size_t num_rays = 0;

    num_rays += tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, num_gc_y), 0.0,
        [&](tbb::blocked_range<size_t> r, size_t num_rays) {  // parallel

    // Loop through 2D-field of grid cells
    //for (size_t i = 0; i < num_gc_y; i++) {  // serial
    for (size_t i=r.begin(); i<r.end(); ++i) {  // parallel
        for (size_t j = 0; j < num_gc_x; j++) {

            size_t lin_ind_gc = lin_ind_2d(num_gc_x, i, j);
            if (mask[lin_ind_gc] == 1) {

            float* norm_tilt = new float[8 * 3];
            float* ray_org = new float[8 * 3];
            float* norm_hori = new float[8 * 3];
            float* surf_enl_fac = new float[8];
            float* sw_dir_cor_ray = new float[8];
            RTCRay8 ray8;

            // Loop through pixels within grid cell (-> process by blocks of 4)
            for (size_t k = (i * pixel_per_gc);
                k < ((i * pixel_per_gc) + pixel_per_gc); k += 2) {
                for (size_t m = (j * pixel_per_gc);
                    m < ((j * pixel_per_gc) + pixel_per_gc); m += 2) {

                    size_t ind_incr_3 = 0;
                    size_t ind_incr_1 = 0;
                    for (size_t k_block = k; k_block < (k + 2); k_block++) {
                    for (size_t m_block = m; m_block < (m + 2); m_block++) {

                    // Loop through two triangles per pixel
                    for (size_t n = 0; n < 2; n++) {

                        //-----------------------------------------------------
                        // Tilted triangle
                        //-----------------------------------------------------

                        size_t ind_tri_0, ind_tri_1, ind_tri_2;
                        func_ptr[n](dem_dim_1,
                            k_block + (pixel_per_gc * offset_gc),
                            m_block + (pixel_per_gc * offset_gc),
                            ind_tri_0, ind_tri_1, ind_tri_2);

                        float vert_0_x = vert_grid[ind_tri_0];
                        float vert_0_y = vert_grid[ind_tri_0 + 1];
                        float vert_0_z = vert_grid[ind_tri_0 + 2];
                        float vert_1_x = vert_grid[ind_tri_1];
                        float vert_1_y = vert_grid[ind_tri_1 + 1];
                        float vert_1_z = vert_grid[ind_tri_1 + 2];
                        float vert_2_x = vert_grid[ind_tri_2];
                        float vert_2_y = vert_grid[ind_tri_2 + 1];
                        float vert_2_z = vert_grid[ind_tri_2 + 2];

                        float cent_x, cent_y, cent_z;
                        triangle_centroid(vert_0_x, vert_0_y, vert_0_z,
                            vert_1_x, vert_1_y, vert_1_z,
                            vert_2_x, vert_2_y, vert_2_z,
                            cent_x, cent_y, cent_z);

                        float norm_tilt_x, norm_tilt_y, norm_tilt_z, area_tilt;
                        triangle_normal_area(vert_0_x, vert_0_y, vert_0_z,
                            vert_1_x, vert_1_y, vert_1_z,
                            vert_2_x, vert_2_y, vert_2_z,
                            norm_tilt_x, norm_tilt_y, norm_tilt_z,
                            area_tilt);
                        norm_tilt[ind_incr_3] = norm_tilt_x;
                        norm_tilt[ind_incr_3 + 1] = norm_tilt_y;
                        norm_tilt[ind_incr_3 + 2] = norm_tilt_z;

                        // Ray origin
                        ray_org[ind_incr_3] = (cent_x
                            + norm_tilt_x * ray_org_elev);
                        ray_org[ind_incr_3 + 1] = (cent_y
                            + norm_tilt_y * ray_org_elev);
                        ray_org[ind_incr_3 + 2] = (cent_z
                            + norm_tilt_z * ray_org_elev);

                        //-----------------------------------------------------
                        // Horizontal triangle
                        //-----------------------------------------------------

                        func_ptr[n](dem_dim_in_1, k_block, m_block,
                            ind_tri_0, ind_tri_1, ind_tri_2);

                        vert_0_x = vert_grid_in[ind_tri_0];
                        vert_0_y = vert_grid_in[ind_tri_0 + 1];
                        vert_0_z = vert_grid_in[ind_tri_0 + 2];
                        vert_1_x = vert_grid_in[ind_tri_1];
                        vert_1_y = vert_grid_in[ind_tri_1 + 1];
                        vert_1_z = vert_grid_in[ind_tri_1 + 2];
                        vert_2_x = vert_grid_in[ind_tri_2];
                        vert_2_y = vert_grid_in[ind_tri_2 + 1];
                        vert_2_z = vert_grid_in[ind_tri_2 + 2];

                        float norm_hori_x, norm_hori_y, norm_hori_z, area_hori;
                        triangle_normal_area(vert_0_x, vert_0_y, vert_0_z,
                            vert_1_x, vert_1_y, vert_1_z,
                            vert_2_x, vert_2_y, vert_2_z,
                            norm_hori_x, norm_hori_y, norm_hori_z,
                            area_hori);
                        norm_hori[ind_incr_3] = norm_hori_x;
                        norm_hori[ind_incr_3 + 1] = norm_hori_y;
                        norm_hori[ind_incr_3 + 2] = norm_hori_z;

                        surf_enl_fac[ind_incr_1] = area_tilt / area_hori;

                        ind_incr_3 = ind_incr_3 + 3;
                        ind_incr_1 = ind_incr_1 + 1;

                    }

                    }
                    }

                    //---------------------------------------------------------
                    // Loop through sun positions and compute correction
                    // factors
                    //---------------------------------------------------------

                    for (size_t o = 0; o < dim_sun_0; o++) {
                        for (size_t p = 0; p < dim_sun_1; p++) {

                            ind_incr_3 = 0;
                            ind_incr_1 = 0;
                            unsigned int num_rays_gc = 0;
                            int valid8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                            // 0: invalid

                            size_t ind_lin_sun
                                = lin_ind_3d(dim_sun_1, 3, o, p, 0);

                            //-------------------------------------------------
                            // Compute correction factors (I)
                            //-------------------------------------------------

                            for (size_t q = 0; q < 8; q++) {

                                // Compute sun unit vector
                                float sun_x = (sun_pos[ind_lin_sun]
                                    - ray_org[ind_incr_3]);
                                float sun_y = (sun_pos[ind_lin_sun + 1]
                                    - ray_org[ind_incr_3 + 1]);
                                float sun_z = (sun_pos[ind_lin_sun + 2]
                                    - ray_org[ind_incr_3 + 2]);
                                vec_unit(sun_x, sun_y, sun_z);

                                // Check for self-shadowing (Earth)
                                float dot_prod_hs
                                    = (norm_hori[ind_incr_3] * sun_x
                                    + norm_hori[ind_incr_3 + 1] * sun_y
                                    + norm_hori[ind_incr_3 + 2] * sun_z);
                                if (dot_prod_hs <= dot_prod_min) {
                                    ind_incr_3 = ind_incr_3 + 3;
                                    ind_incr_1 = ind_incr_1 + 1;
                                    continue;  // sw_dir_cor += 0.0
                                }

                                // Check for self-shadowing (triangle)
                                float dot_prod_ts
                                    = norm_tilt[ind_incr_3] * sun_x
                                    + norm_tilt[ind_incr_3 + 1] * sun_y
                                    + norm_tilt[ind_incr_3 + 2] * sun_z;
                                if (dot_prod_ts <= 0.0) {
                                    ind_incr_3 = ind_incr_3 + 3;
                                    ind_incr_1 = ind_incr_1 + 1;
                                    continue;  // sw_dir_cor += 0.0
                                }

                                // Add ray
                                ray8.org_x[num_rays_gc]
                                    = ray_org[ind_incr_3];
                                ray8.org_y[num_rays_gc]
                                    = ray_org[ind_incr_3 + 1];
                                ray8.org_z[num_rays_gc]
                                    = ray_org[ind_incr_3 + 2];
                                ray8.tnear[num_rays_gc] = 0.0;
                                ray8.dir_x[num_rays_gc] = sun_x;
                                ray8.dir_y[num_rays_gc] = sun_y;
                                ray8.dir_z[num_rays_gc] = sun_z;
                                ray8.tfar[num_rays_gc] = dist_search;
                                // std::numeric_limits<float>::infinity();
                                ray8.id[num_rays_gc] = num_rays_gc;
                                valid8[num_rays_gc] = -1; // -1: valid

                                sw_dir_cor_ray[num_rays_gc] =
                                    std::min(((dot_prod_ts / dot_prod_hs)
                                    * surf_enl_fac[ind_incr_1]),
                                    sw_dir_cor_max);
                                num_rays_gc = num_rays_gc + 1;

                                ind_incr_3 = ind_incr_3 + 3;
                                ind_incr_1 = ind_incr_1 + 1;

                            }

                            //-------------------------------------------------
                            // Compute correction factors (II)
                            //-------------------------------------------------

                            struct RTCIntersectContext context;
                            rtcInitIntersectContext(&context);
                            context.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;

                            // Intersect rays with scene
                            rtcOccluded8(valid8, scene, &context,
                                (RTCRay8*)&ray8);
                            num_rays += num_rays_gc;

                            float sw_dir_cor_agg = 0.0;
                            for (size_t n = 0; n < num_rays_gc; n++) {
                                if (ray8.tfar[n] > 0.0) {
                                // no intersection -> 'tfar' is not updated;
                                // otherwise 'tfar' = -inf
                                sw_dir_cor_agg = sw_dir_cor_agg
                                    + sw_dir_cor_ray[n];
                                }  // else: sw_dir_cor += 0.0
                            }

                            size_t ind_lin_cor = lin_ind_4d(num_gc_x,
                                dim_sun_0, dim_sun_1, i, j, o, p);
                            sw_dir_cor[ind_lin_cor] = sw_dir_cor[ind_lin_cor]
                                + sw_dir_cor_agg;

                        }
                    }

                }
            }

            delete[] norm_tilt;
            delete[] ray_org;
            delete[] norm_hori;
            delete[] surf_enl_fac;
            delete[] sw_dir_cor_ray;
            
            } else {

                size_t ind_lin = lin_ind_4d(num_gc_x, dim_sun_0, dim_sun_1,
                    i, j, 0, 0);
                for (size_t k = 0; k < (dim_sun_0 * dim_sun_1) ; k++) {
                    sw_dir_cor[ind_lin + k] = NAN;
                }

            }

        }
    }

    return num_rays;  // parallel
    }, std::plus<size_t>());  // parallel

    auto end_ray = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_ray = (end_ray - start_ray);
    cout << "Ray tracing time: " << time_ray.count() << " s" << endl;
    cout << "Number of rays shot: " << num_rays << endl;
    float frac_ray = (float)num_rays /
        ((float)num_tri * (float)dim_sun_0 * (float)dim_sun_1);
    cout << "Fraction of rays required: " << frac_ray << endl;

    // Divide accumulated values by number of triangles within grid cell
    float num_tri_per_gc = pixel_per_gc * pixel_per_gc * 2.0;
    size_t num_elem = (num_gc_y * num_gc_x * dim_sun_0 * dim_sun_1);
    for (size_t i = 0; i < num_elem; i++) {
        sw_dir_cor[i] /= num_tri_per_gc;
    }

    // Release resources allocated through Embree
    rtcReleaseScene(scene);
    rtcReleaseDevice(device);

    auto end_tot = std::chrono::high_resolution_clock::now();
    time = end_tot - start_ini;
    cout << "Total run time: " << time.count() << " s" << endl;

    //-------------------------------------------------------------------------

    cout << "--------------------------------------------------------" << endl;

}
