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
inline double deg2rad(double ang) {
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
inline double rad2deg(double ang) {
    /* Parameters
       ----------
       ang: angle [radian]

       Returns
       ----------
       ang: angle [degree]*/
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
inline void vec_unit(double &v_x, double &v_y, double &v_z) {
    /* Parameters
       ----------
       v_x: x-component of vector [arbitrary]
       v_y: y-component of vector [arbitrary]
       v_z: z-component of vector [arbitrary]
    */
    double mag = sqrt(v_x * v_x + v_y * v_y + v_z * v_z);
    v_x = v_x / mag;
    v_y = v_y / mag;
    v_z = v_z / mag;
}

// Cross product
inline void cross_prod(double a_x, double a_y, double a_z,
    double b_x, double b_y, double b_z,
    double &c_x, double &c_y, double &c_z) {
    /* Parameters
       ----------
       a_x: x-component of vector a [arbitrary]
       a_y: y-component of vector a [arbitrary]
       a_z: z-component of vector a [arbitrary]
       b_x: x-component of vector b [arbitrary]
       b_y: y-component of vector b [arbitrary]
       b_z: z-component of vector b [arbitrary]
       c_x: x-component of vector c [arbitrary]
       c_y: y-component of vector c [arbitrary]
       c_z: z-component of vector c [arbitrary]
    */
    c_x = a_y * b_z - a_z * b_y;
    c_y = a_z * b_x - a_x * b_z;
    c_z = a_x * b_y - a_y * b_x;
}

// Matrix-vector multiplication
inline void mat_vec_mult(double (&mat)[3][3], double (&vec)[3],
    double (&vec_res)[3]) {
    /* Parameters
       ----------
       mat: matrix with 3 x 3 elements [arbitrary]
       vec: vector with 3 elements [arbitrary]
       vec_res: resulting vector with 3 elements [arbitrary]
    */
    vec_res[0] = mat[0][0] * vec[0] + mat[0][1] * vec[1] + mat[0][2] * vec[2];
    vec_res[1] = mat[1][0] * vec[0] + mat[1][1] * vec[1] + mat[1][2] * vec[2];
    vec_res[2] = mat[2][0] * vec[0] + mat[2][1] * vec[1] + mat[2][2] * vec[2];

}

// ----------------------------------------------------------------------------
// Triangle operations
// ----------------------------------------------------------------------------

// Triangle surface normal and area
inline void triangle_normal_area(
    double &vert_0_x, double &vert_0_y, double &vert_0_z,
    double &vert_1_x, double &vert_1_y, double &vert_1_z,
    double &vert_2_x, double &vert_2_y, double &vert_2_z,
    double &norm_x, double &norm_y, double &norm_z,
    double &area) {
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
    double a_x = vert_2_x - vert_1_x;
    double a_y = vert_2_y - vert_1_y;
    double a_z = vert_2_z - vert_1_z;
    double b_x = vert_0_x - vert_1_x;
    double b_y = vert_0_y - vert_1_y;
    double b_z = vert_0_z - vert_1_z;

    norm_x = a_y * b_z - a_z * b_y;
    norm_y = a_z * b_x - a_x * b_z;
    norm_z = a_x * b_y - a_y * b_x;

    double mag = sqrt(norm_x * norm_x + norm_y * norm_y + norm_z * norm_z);
    norm_x = norm_x / mag;
    norm_y = norm_y / mag;
    norm_z = norm_z / mag;

    area = mag / 2.0;
}

// Triangle centroid
inline void triangle_centroid(
    double &vert_0_x, double &vert_0_y, double &vert_0_z,
    double &vert_1_x, double &vert_1_y, double &vert_1_z,
    double &vert_2_x, double &vert_2_y, double &vert_2_z,
    double &cent_x, double &cent_y, double &cent_z) {
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
// Ray casting
//#############################################################################

bool castRay_occluded1(RTCScene scene, float ox, float oy, float oz, float dx,
    float dy, float dz, float dist_search) {

    // Intersect context
    struct RTCIntersectContext context;
    rtcInitIntersectContext(&context);

    // Ray structure
    struct RTCRay ray;
    ray.org_x = ox;
    ray.org_y = oy;
    ray.org_z = oz;
    ray.dir_x = dx;
    ray.dir_y = dy;
    ray.dir_z = dz;
    ray.tnear = 0.0;
    //ray.tfar = std::numeric_limits<float>::infinity();
    ray.tfar = dist_search;
    //ray.mask = -1;
    //ray.flags = 0;

    // Intersect ray with scene
    rtcOccluded1(scene, &context, &ray);

    return (ray.tfar < 0.0);

}

//-----------------------------------------------------------------------------

bool castRay_intersect1(RTCScene scene, float ox, float oy, float oz, float dx,
	float dy, float dz, float dist_search, float &dist) {

    // Intersect context
    struct RTCIntersectContext context;
    rtcInitIntersectContext(&context);

    // Ray hit structure
    struct RTCRayHit rayhit;
    rayhit.ray.org_x = ox;
    rayhit.ray.org_y = oy;
    rayhit.ray.org_z = oz;
    rayhit.ray.dir_x = dx;
    rayhit.ray.dir_y = dy;
    rayhit.ray.dir_z = dz;
    rayhit.ray.tnear = 0.0;
    //rayhit.ray.tfar = std::numeric_limits<float>::infinity();
    rayhit.ray.tfar = dist_search;
    //rayhit.ray.mask = -1;
    //rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    // Intersect ray with scene
    rtcIntersect1(scene, &context, &rayhit);
    dist = rayhit.ray.tfar;

    return (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);

}

//#############################################################################
// Horizon detection algorithms
//#############################################################################

//-----------------------------------------------------------------------------
// Discrete sampling
//-----------------------------------------------------------------------------

void ray_discrete_sampling(float ray_org_x, float ray_org_y, float ray_org_z,
    size_t azim_num, double hori_acc, float dist_search,
    double elev_ang_low_lim, double elev_ang_up_lim, int elev_num,
    RTCScene scene, size_t &num_rays, double* horizon,
    double* azim_sin, double* azim_cos, double* elev_ang,
    double* elev_cos, double* elev_sin, double (&rot_inv)[3][3]) {

    for (size_t k = 0; k < azim_num; k++) {

        int ind_elev = 0;
        int ind_elev_prev = 0;
        bool hit = true;
        while (hit) {

            ind_elev_prev = ind_elev;
            ind_elev = min(ind_elev + 10, elev_num - 1);
            double ray[3] = {elev_cos[ind_elev] * azim_sin[k],
                            elev_cos[ind_elev] * azim_cos[k],
                            elev_sin[ind_elev]};
            double ray_rot[3];
            mat_vec_mult(rot_inv, ray, ray_rot);
            hit = castRay_occluded1(scene,
                ray_org_x, ray_org_y, ray_org_z,
                (float)ray_rot[0], (float)ray_rot[1], (float)ray_rot[2],
                dist_search);
            num_rays += 1;

        }
        horizon[k] = (elev_ang[ind_elev_prev] + elev_ang[ind_elev]) / 2.0;

    }

}

//-----------------------------------------------------------------------------
// Binary search
//-----------------------------------------------------------------------------

void ray_binary_search(float ray_org_x, float ray_org_y, float ray_org_z,
    size_t azim_num, double hori_acc, float dist_search,
    double elev_ang_low_lim, double elev_ang_up_lim, int elev_num,
    RTCScene scene, size_t &num_rays, double* horizon,
    double* azim_sin, double* azim_cos, double* elev_ang,
    double* elev_cos, double* elev_sin, double (&rot_inv)[3][3]) {

    for (size_t k = 0; k < azim_num; k++) {

        double lim_up = elev_ang_up_lim;
        double lim_low = elev_ang_low_lim;
        double elev_samp = (lim_up + lim_low) / 2.0;
        int ind_elev = ((int)round((elev_samp - elev_ang_low_lim)
            / (hori_acc / 5.0)));

        while (max(lim_up - elev_ang[ind_elev],
            elev_ang[ind_elev] - lim_low) > hori_acc) {

            double ray[3] = {elev_cos[ind_elev] * azim_sin[k],
                            elev_cos[ind_elev] * azim_cos[k],
                            elev_sin[ind_elev]};
            double ray_rot[3];
            mat_vec_mult(rot_inv, ray, ray_rot);
            bool hit = castRay_occluded1(scene,
                ray_org_x, ray_org_y, ray_org_z,
                (float)ray_rot[0], (float)ray_rot[1], (float)ray_rot[2],
                dist_search);
            num_rays += 1;

            if (hit) {
                lim_low = elev_ang[ind_elev];
            } else {
                lim_up = elev_ang[ind_elev];
            }
            elev_samp = (lim_up + lim_low) / 2.0;
            ind_elev = ((int)round((elev_samp - elev_ang_low_lim)
                / (hori_acc / 5.0)));

        }
        horizon[k] = elev_samp;

    }

}

//-----------------------------------------------------------------------------
// Guess horizon from previous azimuth direction
//-----------------------------------------------------------------------------

void ray_guess_const(float ray_org_x, float ray_org_y, float ray_org_z,
    size_t azim_num, double hori_acc, float dist_search,
    double elev_ang_low_lim, double elev_ang_up_lim, int elev_num,
    RTCScene scene, size_t &num_rays, double* horizon,
    double* azim_sin, double* azim_cos, double* elev_ang,
    double* elev_cos, double* elev_sin, double (&rot_inv)[3][3]) {

    // ------------------------------------------------------------------------
    // First azimuth direction (binary search)
    // ------------------------------------------------------------------------

    double lim_up = elev_ang_up_lim;
    double lim_low = elev_ang_low_lim;
    double elev_samp = (lim_up + lim_low) / 2.0;
    int ind_elev = ((int)round((elev_samp - elev_ang_low_lim)
        / (hori_acc / 5.0)));

    while (max(lim_up - elev_ang[ind_elev],
        elev_ang[ind_elev] - lim_low) > hori_acc) {

        double ray[3] = {elev_cos[ind_elev] * azim_sin[0],
                        elev_cos[ind_elev] * azim_cos[0],
                        elev_sin[ind_elev]};
        double ray_rot[3];
        mat_vec_mult(rot_inv, ray, ray_rot);
        bool hit = castRay_occluded1(scene,
            ray_org_x, ray_org_y, ray_org_z,
            (float)ray_rot[0], (float)ray_rot[1], (float)ray_rot[2],
            dist_search);
        num_rays += 1;

        if (hit) {
            lim_low = elev_ang[ind_elev];
        } else {
            lim_up = elev_ang[ind_elev];
        }
        elev_samp = (lim_up + lim_low) / 2.0;
        ind_elev = ((int)round((elev_samp - elev_ang_low_lim)
            / (hori_acc / 5.0)));

    }

    horizon[0] = elev_samp;
    int ind_elev_prev_azim = ind_elev;

    // ------------------------------------------------------------------------
    // Remaining azimuth directions (guess horizon from previous
    // azimuth direction)
    // ------------------------------------------------------------------------

    for (size_t k = 1; k < azim_num; k++) {

        // Move upwards
        ind_elev = max(ind_elev_prev_azim - 5, 0);
        int ind_elev_prev = 0;
        bool hit = true;
        int count = 0;
        while (hit) {

            ind_elev_prev = ind_elev;
            ind_elev = min(ind_elev + 10, elev_num - 1);
            double ray[3] = {elev_cos[ind_elev] * azim_sin[k],
                            elev_cos[ind_elev] * azim_cos[k],
                            elev_sin[ind_elev]};
            double ray_rot[3];
            mat_vec_mult(rot_inv, ray, ray_rot);
            hit = castRay_occluded1(scene,
                ray_org_x, ray_org_y, ray_org_z,
                (float)ray_rot[0], (float)ray_rot[1], (float)ray_rot[2],
                dist_search);
            num_rays += 1;
            count += 1;

        }

        if (count > 1) {

            elev_samp = (elev_ang[ind_elev_prev] + elev_ang[ind_elev]) / 2.0;
            ind_elev = ((int)round((elev_samp - elev_ang_low_lim)
                / (hori_acc / 5.0)));
            horizon[k] = elev_ang[ind_elev];
            ind_elev_prev_azim = ind_elev;
            continue;

        }

        // Move downwards
        ind_elev = min(ind_elev_prev_azim + 5, elev_num - 1);
        hit = false;
        while (!hit) {

            ind_elev_prev = ind_elev;
            ind_elev = max(ind_elev - 10, 0);
            double ray[3] = {elev_cos[ind_elev] * azim_sin[k],
                            elev_cos[ind_elev] * azim_cos[k],
                            elev_sin[ind_elev]};
            double ray_rot[3];
            mat_vec_mult(rot_inv, ray, ray_rot);
            hit = castRay_occluded1(scene,
                ray_org_x, ray_org_y, ray_org_z,
                (float)ray_rot[0], (float)ray_rot[1], (float)ray_rot[2],
                dist_search);
            num_rays += 1;

        }

        elev_samp = (elev_ang[ind_elev_prev] + elev_ang[ind_elev]) / 2.0;
        ind_elev = ((int)round((elev_samp - elev_ang_low_lim)
            / (hori_acc / 5.0)));
        horizon[k] = elev_ang[ind_elev];
        ind_elev_prev_azim = ind_elev;

    }

}

//-----------------------------------------------------------------------------
// Declare function pointer and assign function
//-----------------------------------------------------------------------------

void (*function_pointer)(float ray_org_x, float ray_org_y, float ray_org_z,
    size_t azim_num, double hori_acc, float dist_search,
    double elev_ang_low_lim, double elev_ang_up_lim, int elev_num,
    RTCScene scene, size_t &num_rays, double* horizon,
    double* azim_sin, double* azim_cos, double* elev_ang,
    double* elev_cos, double* elev_sin, double (&rot_inv)[3][3]);

//#############################################################################
// Sample hemisphere
//#############################################################################

void sample_hemisphere(float ray_org_x, float ray_org_y, float ray_org_z,
    size_t azim_num, size_t elev_num, float dist_search,
    RTCScene scene, size_t &num_rays, double &dist_mean,
    double* azim_sin, double* azim_cos,
    double* elev_cos, double* elev_sin, double (&rot_inv)[3][3]) {

    int num_hit = 0;
    for (size_t k = 0; k < azim_num; k++) {

        int ind_elev = 0;
        bool hit = true;
        while (hit) {

            double ray[3] = {elev_cos[ind_elev] * azim_sin[k],
                             elev_cos[ind_elev] * azim_cos[k],
                             elev_sin[ind_elev]};
            double ray_rot[3];
            mat_vec_mult(rot_inv, ray, ray_rot);
            float dist;
            hit = castRay_intersect1(scene,
                ray_org_x, ray_org_y, ray_org_z,
                (float)ray_rot[0], (float)ray_rot[1], (float)ray_rot[2],
                dist_search, dist);
            num_rays += 1;
            ind_elev = ind_elev + 1;
            if (hit) {
                dist_mean = dist_mean + dist;
                num_hit = num_hit + 1;
            }

        }

    }
    dist_mean = dist_mean / (double)num_hit;

}

//#############################################################################
// Main functions
//#############################################################################

//-----------------------------------------------------------------------------
// Compute sky view factor
//-----------------------------------------------------------------------------

void sky_view_factor_comp(
    float* vert_grid,
    int dem_dim_0, int dem_dim_1,
    float* vert_grid_in,
    int dem_dim_in_0, int dem_dim_in_1,
    double* north_pole,
    double* sky_view_factor,
    double* area_increase_factor,
    double* sky_view_area_factor,
    double* slope,
    double* aspect,
    int pixel_per_gc,
    int offset_gc,
    uint8_t* mask,
    float dist_search,
    int hori_azim_num,
    double hori_acc,
    char* ray_algorithm,
    double elev_ang_low_lim,
    char* geom_type) {

    cout << "--------------------------------------------------------" << endl;
    cout << "Compute sky view factor " << endl;
    cout << "--------------------------------------------------------" << endl;

    // Hard-coded settings
    double ray_org_elev = 0.1;
    // value to elevate ray origin (-> avoids potential issue with numerical
    // imprecision / truncation) [m]
    double elev_ang_up_lim = 89.98;
    // upper limit for elevation angle [degree]

    // Number of grid cells
    int num_gc_y = (dem_dim_in_0 - 1) / pixel_per_gc;
    int num_gc_x = (dem_dim_in_1 - 1) / pixel_per_gc;
    cout << "Number of grid cells in y-direction: " << num_gc_y << endl;
    cout << "Number of grid cells in x-direction: " << num_gc_x << endl;

    // Number of triangles
    int num_tri = (dem_dim_in_0 - 1) * (dem_dim_in_1 - 1) * 2;
    cout << "Number of triangles: " << num_tri << endl;

    // Unit conversion(s)
    dist_search *= 1000.0;  // [kilometre] to [metre]
    cout << "Search distance: " << dist_search << " m" << endl;
    hori_acc = deg2rad(hori_acc);
    elev_ang_low_lim = deg2rad(elev_ang_low_lim);
    elev_ang_up_lim = deg2rad(elev_ang_up_lim);

    // Select algorithm for horizon detection
    cout << "Horizon detection algorithm: ";
    if (strcmp(ray_algorithm, "discrete_sampling") == 0) {
        cout << "discrete_sampling" << endl;
        function_pointer = ray_discrete_sampling;
    } else if (strcmp(ray_algorithm, "binary_search") == 0) {
        cout << "binary search" << endl;
        function_pointer = ray_binary_search;
    } else if (strcmp(ray_algorithm, "guess_constant") == 0) {
        cout << "guess horizon from previous azimuth direction" << endl;
        function_pointer = ray_guess_const;
    }

    // Initialisation
    auto start_ini = std::chrono::high_resolution_clock::now();
    RTCDevice device = initializeDevice();
    RTCScene scene = initializeScene(device, vert_grid, dem_dim_0, dem_dim_1,
        geom_type);
    auto end_ini = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = end_ini - start_ini;
    cout << "Total initialisation time: " << time.count() << " s" << endl;

    // ------------------------------------------------------------------------
    // Allocate and initialise arrays with evaluated trigonometric functions
    // ------------------------------------------------------------------------

    // Azimuth angles (allocate on stack)
    double azim_sin[hori_azim_num];
    double azim_cos[hori_azim_num];
    double ang;
    for (int i = 0; i < hori_azim_num; i++) {
        ang = ((2 * M_PI) / hori_azim_num * i);
        azim_sin[i] = sin(ang);
        azim_cos[i] = cos(ang);
    }

    // Elevation angles (allocate on stack)
    int elev_num = ((int)ceil((elev_ang_up_lim - elev_ang_low_lim)
        / (hori_acc / 5.0)) + 1);
    double elev_ang[elev_num];
    double elev_sin[elev_num];
    double elev_cos[elev_num];
    for (int i = 0; i < elev_num; i++) {
        ang = elev_ang_up_lim - (hori_acc / 5.0) * i;
        elev_ang[elev_num - i - 1] = ang;
        elev_sin[elev_num - i - 1] = sin(ang);
        elev_cos[elev_num - i - 1] = cos(ang);
    }
    double azim_spac = (2.0 * M_PI) / (double)hori_azim_num;

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

            double* horizon = new double[hori_azim_num];
            double* tilt_gc = new double[3] {0.0, 0.0, 0.0};

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

                        double vert_0_x = (double)vert_grid[ind_tri_0];
                        double vert_0_y = (double)vert_grid[ind_tri_0 + 1];
                        double vert_0_z = (double)vert_grid[ind_tri_0 + 2];
                        double vert_1_x = (double)vert_grid[ind_tri_1];
                        double vert_1_y = (double)vert_grid[ind_tri_1 + 1];
                        double vert_1_z = (double)vert_grid[ind_tri_1 + 2];
                        double vert_2_x = (double)vert_grid[ind_tri_2];
                        double vert_2_y = (double)vert_grid[ind_tri_2 + 1];
                        double vert_2_z = (double)vert_grid[ind_tri_2 + 2];

                        double cent_x, cent_y, cent_z;
                        triangle_centroid(vert_0_x, vert_0_y, vert_0_z,
                            vert_1_x, vert_1_y, vert_1_z,
                            vert_2_x, vert_2_y, vert_2_z,
                            cent_x, cent_y, cent_z);

                        double norm_tilt_x, norm_tilt_y, norm_tilt_z,
                            area_tilt;
                        triangle_normal_area(vert_0_x, vert_0_y, vert_0_z,
                            vert_1_x, vert_1_y, vert_1_z,
                            vert_2_x, vert_2_y, vert_2_z,
                            norm_tilt_x, norm_tilt_y, norm_tilt_z,
                            area_tilt);

                        // Ray origin
                        double ray_org_x = (cent_x
                            + norm_tilt_x * ray_org_elev);
                        double ray_org_y = (cent_y
                            + norm_tilt_y * ray_org_elev);
                        double ray_org_z = (cent_z
                            + norm_tilt_z * ray_org_elev);

                        //-----------------------------------------------------
                        // Horizontal triangle
                        //-----------------------------------------------------

                        func_ptr[n](dem_dim_in_1, k, m,
                            ind_tri_0, ind_tri_1, ind_tri_2);

                        vert_0_x = (double)vert_grid_in[ind_tri_0];
                        vert_0_y = (double)vert_grid_in[ind_tri_0 + 1];
                        vert_0_z = (double)vert_grid_in[ind_tri_0 + 2];
                        vert_1_x = (double)vert_grid_in[ind_tri_1];
                        vert_1_y = (double)vert_grid_in[ind_tri_1 + 1];
                        vert_1_z = (double)vert_grid_in[ind_tri_1 + 2];
                        vert_2_x = (double)vert_grid_in[ind_tri_2];
                        vert_2_y = (double)vert_grid_in[ind_tri_2 + 1];
                        vert_2_z = (double)vert_grid_in[ind_tri_2 + 2];

                        double norm_hori_x, norm_hori_y, norm_hori_z,
                            area_hori;
                        triangle_normal_area(vert_0_x, vert_0_y, vert_0_z,
                            vert_1_x, vert_1_y, vert_1_z,
                            vert_2_x, vert_2_y, vert_2_z,
                            norm_hori_x, norm_hori_y, norm_hori_z,
                        area_hori);

                        double surf_enl_fac = area_tilt / area_hori;

                        //-----------------------------------------------------
                        // Compute horizon in local ENU coordinate system
                        //-----------------------------------------------------

                        // Vector to North Pole (and orthogonal to 'norm_hori')
                        double north_x = (north_pole[0] - cent_x);
                        double north_y = (north_pole[1] - cent_y);
                        double north_z = (north_pole[2] - cent_z);
                        double dot_prod = ((north_x * norm_hori_x)
                                         + (north_y * norm_hori_y)
                                         + (north_z * norm_hori_z));
                        north_x -= dot_prod * norm_hori_x;
                        north_y -= dot_prod * norm_hori_y;
                        north_z -= dot_prod * norm_hori_z;
                        vec_unit(north_x, north_y, north_z);

                        double east_x, east_y, east_z;
                        cross_prod(north_x, north_y, north_z,
                            norm_hori_x, norm_hori_y, norm_hori_z,
                            east_x, east_y, east_z);
                        double rot_inv[3][3] =
                            {{east_x, north_x, norm_hori_x},
                             {east_y, north_y, norm_hori_y},
                             {east_z, north_z, norm_hori_z}};

                        function_pointer(
                            (float)ray_org_x, (float)ray_org_y,
                            (float)ray_org_z,
                            hori_azim_num, hori_acc, dist_search,
                            elev_ang_low_lim, elev_ang_up_lim, elev_num,
                            scene, num_rays, &horizon[0],
                            azim_sin, azim_cos, elev_ang,
                            elev_cos, elev_sin, rot_inv);
                        // values that are no used for operations and are only
                        // passed are already converted to 'float' here

                        //-----------------------------------------------------
                        // Compute sky view factor
                        //-----------------------------------------------------

                        // Rotate tilt vector from global to local ENU
                        // coordinate system
                        double rot[3][3] = {{east_x, east_y, east_z},
                                            {north_x, north_y, north_z},
                                            {norm_hori_x, norm_hori_y,
                                             norm_hori_z}};
                        double tilt_global[3] = {norm_tilt_x, norm_tilt_y,
                                                 norm_tilt_z};
                        double tilt_local[3];
                        mat_vec_mult(rot, tilt_global, tilt_local);
                        tilt_gc[0] = tilt_gc[0] + tilt_local[0];
                        tilt_gc[1] = tilt_gc[1] + tilt_local[1];
                        tilt_gc[2] = tilt_gc[2] + tilt_local[2];

                        // Compute sky view factor
                        double agg = 0.0;
                        for (size_t o = 0; o < hori_azim_num; o++) {

                            // Compute plane-sphere intersection
                            double hori_plane = atan(-azim_sin[o]
                                * tilt_local[0] / tilt_local[2]
                                - azim_cos[o] * tilt_local[1] / tilt_local[2]);
                            double hori_elev;
                            if (horizon[o] >= hori_plane) {
                                hori_elev = horizon[o];
                            } else {
                                hori_elev =  hori_plane;
                            }

                            // Compute inner integral
                            agg = agg + ((tilt_local[0] * azim_sin[o]
                                + tilt_local[1] * azim_cos[o]) * ((M_PI / 2.0)
                                - hori_elev - (sin(2.0 * hori_elev) / 2.0))
                                + tilt_local[2] * pow(cos(hori_elev), 2));

                        }

                        sky_view_factor[lin_ind_gc]
                            = sky_view_factor[lin_ind_gc]
                            + (azim_spac / (2.0 * M_PI)) * agg;

                        area_increase_factor[lin_ind_gc]
                            = area_increase_factor[lin_ind_gc]
                            + surf_enl_fac;

                        sky_view_area_factor[lin_ind_gc]
                            = sky_view_area_factor[lin_ind_gc]
                            + ((azim_spac / (2.0 * M_PI)) * agg)
                            * surf_enl_fac;

                    }

                }
            }

            // Compute mean grid cell slope and aspect
            vec_unit(tilt_gc[0], tilt_gc[1], tilt_gc[2]);
            slope[lin_ind_gc] = rad2deg(acos(tilt_gc[2]));
            double aspect_temp = atan2(tilt_gc[0], tilt_gc[1]);
            if (aspect_temp < 0.0) {
                aspect_temp += 2.0 * M_PI;
            }
            aspect[lin_ind_gc] = rad2deg(aspect_temp);

            delete[] horizon;
            delete[] tilt_gc;

            } else {

                sky_view_factor[lin_ind_gc] = NAN;
                area_increase_factor[lin_ind_gc] = NAN;
                sky_view_area_factor[lin_ind_gc] = NAN;
                slope[lin_ind_gc] = NAN;
                aspect[lin_ind_gc] = NAN;

            }

        }
    }

    return num_rays;  // parallel
    }, std::plus<size_t>());  // parallel

    auto end_ray = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_ray = (end_ray - start_ray);
    cout << "Ray tracing time: " << time_ray.count() << " s" << endl;

    // Print number of rays needed for location and azimuth direction
    cout << "Number of rays shot: " << num_rays << endl;
    double gc_proc = 0;
    for (size_t i = 0; i < (num_gc_y * num_gc_x); i++) {
        if (mask[i] == 1) {
            gc_proc += 1;
        }
    }
    double tri_proc = (gc_proc * (double)(pixel_per_gc * pixel_per_gc) * 2.0)
        * (double)hori_azim_num;
    double ratio = (double)num_rays / (double)tri_proc;
    printf("Average number of rays per location and azimuth: %.2f \n", ratio);

    // Divide accumulated values by number of triangles within grid cell
    double num_tri_per_gc = pixel_per_gc * pixel_per_gc * 2.0;
    size_t num_elem = (num_gc_y * num_gc_x);
    for (size_t i = 0; i < num_elem; i++) {
        sky_view_factor[i] /= num_tri_per_gc;
        area_increase_factor[i] /= num_tri_per_gc;
        sky_view_area_factor[i] /= num_tri_per_gc;
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
// Compute sky view factor and SW_dir correction factor
//-----------------------------------------------------------------------------

void sky_view_factor_sw_dir_cor_comp(
    float* vert_grid,
    int dem_dim_0, int dem_dim_1,
    float* vert_grid_in,
    int dem_dim_in_0, int dem_dim_in_1,
    double* north_pole,
    double* sun_pos,
    int dim_sun_0, int dim_sun_1,
    float* sw_dir_cor,
    double* sky_view_factor,
    double* area_increase_factor,
    double* sky_view_area_factor,
    double* slope,
    double* aspect,
    int pixel_per_gc,
    int offset_gc,
    uint8_t* mask,
    float dist_search,
    int hori_azim_num,
    double hori_acc,
    char* ray_algorithm,
    double elev_ang_low_lim,
    char* geom_type,
    double sw_dir_cor_max,
    double ang_max) {

    cout << "--------------------------------------------------------" << endl;
    cout << "Compute sky view factor and SW_dir correction factor" << endl;
    cout << "--------------------------------------------------------" << endl;

    // Hard-coded settings
    double ray_org_elev = 0.1;
    // value to elevate ray origin (-> avoids potential issue with numerical
    // imprecision / truncation) [m]
    double elev_ang_up_lim = 89.98;
    // upper limit for elevation angle [degree]

    // Number of grid cells
    int num_gc_y = (dem_dim_in_0 - 1) / pixel_per_gc;
    int num_gc_x = (dem_dim_in_1 - 1) / pixel_per_gc;
    cout << "Number of grid cells in y-direction: " << num_gc_y << endl;
    cout << "Number of grid cells in x-direction: " << num_gc_x << endl;

    // Number of triangles
    int num_tri = (dem_dim_in_0 - 1) * (dem_dim_in_1 - 1) * 2;
    cout << "Number of triangles: " << num_tri << endl;

    // Unit conversion(s)
    double dot_prod_min = cos(deg2rad(ang_max));
    dist_search *= 1000.0;  // [kilometre] to [metre]
    cout << "Search distance: " << dist_search << " m" << endl;
    hori_acc = deg2rad(hori_acc);
    elev_ang_low_lim = deg2rad(elev_ang_low_lim);
    elev_ang_up_lim = deg2rad(elev_ang_up_lim);

    // Select algorithm for horizon detection
    cout << "Horizon detection algorithm: ";
    if (strcmp(ray_algorithm, "discrete_sampling") == 0) {
        cout << "discrete_sampling" << endl;
        function_pointer = ray_discrete_sampling;
    } else if (strcmp(ray_algorithm, "binary_search") == 0) {
        cout << "binary search" << endl;
        function_pointer = ray_binary_search;
    } else if (strcmp(ray_algorithm, "guess_constant") == 0) {
        cout << "guess horizon from previous azimuth direction" << endl;
        function_pointer = ray_guess_const;
    }

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

    // ------------------------------------------------------------------------
    // Allocate and initialise arrays with evaluated trigonometric functions
    // ------------------------------------------------------------------------

    // Azimuth angles (allocate on stack)
    double azim_sin[hori_azim_num];
    double azim_cos[hori_azim_num];
    double ang;
    for (int i = 0; i < hori_azim_num; i++) {
        ang = ((2 * M_PI) / hori_azim_num * i);
        azim_sin[i] = sin(ang);
        azim_cos[i] = cos(ang);
    }

    // Elevation angles (allocate on stack)
    int elev_num = ((int)ceil((elev_ang_up_lim - elev_ang_low_lim)
        / (hori_acc / 5.0)) + 1);
    double elev_ang[elev_num];
    double elev_sin[elev_num];
    double elev_cos[elev_num];
    for (int i = 0; i < elev_num; i++) {
        ang = elev_ang_up_lim - (hori_acc / 5.0) * i;
        elev_ang[elev_num - i - 1] = ang;
        elev_sin[elev_num - i - 1] = sin(ang);
        elev_cos[elev_num - i - 1] = cos(ang);
    }
    double azim_spac = (2.0 * M_PI) / (double)hori_azim_num;

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

            double* horizon = new double[hori_azim_num + 1];
            double* horizon_sin = new double[hori_azim_num + 1];
            // save horizon in 'periodical' array for interpolation
            double* tilt_gc = new double[3] {0.0, 0.0, 0.0};

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

                        double vert_0_x = (double)vert_grid[ind_tri_0];
                        double vert_0_y = (double)vert_grid[ind_tri_0 + 1];
                        double vert_0_z = (double)vert_grid[ind_tri_0 + 2];
                        double vert_1_x = (double)vert_grid[ind_tri_1];
                        double vert_1_y = (double)vert_grid[ind_tri_1 + 1];
                        double vert_1_z = (double)vert_grid[ind_tri_1 + 2];
                        double vert_2_x = (double)vert_grid[ind_tri_2];
                        double vert_2_y = (double)vert_grid[ind_tri_2 + 1];
                        double vert_2_z = (double)vert_grid[ind_tri_2 + 2];

                        double cent_x, cent_y, cent_z;
                        triangle_centroid(vert_0_x, vert_0_y, vert_0_z,
                            vert_1_x, vert_1_y, vert_1_z,
                            vert_2_x, vert_2_y, vert_2_z,
                            cent_x, cent_y, cent_z);

                        double norm_tilt_x, norm_tilt_y, norm_tilt_z,
                            area_tilt;
                        triangle_normal_area(vert_0_x, vert_0_y, vert_0_z,
                            vert_1_x, vert_1_y, vert_1_z,
                            vert_2_x, vert_2_y, vert_2_z,
                            norm_tilt_x, norm_tilt_y, norm_tilt_z,
                            area_tilt);

                        // Ray origin
                        double ray_org_x = (cent_x
                            + norm_tilt_x * ray_org_elev);
                        double ray_org_y = (cent_y
                            + norm_tilt_y * ray_org_elev);
                        double ray_org_z = (cent_z
                            + norm_tilt_z * ray_org_elev);

                        //-----------------------------------------------------
                        // Horizontal triangle
                        //-----------------------------------------------------

                        func_ptr[n](dem_dim_in_1, k, m,
                            ind_tri_0, ind_tri_1, ind_tri_2);

                        vert_0_x = (double)vert_grid_in[ind_tri_0];
                        vert_0_y = (double)vert_grid_in[ind_tri_0 + 1];
                        vert_0_z = (double)vert_grid_in[ind_tri_0 + 2];
                        vert_1_x = (double)vert_grid_in[ind_tri_1];
                        vert_1_y = (double)vert_grid_in[ind_tri_1 + 1];
                        vert_1_z = (double)vert_grid_in[ind_tri_1 + 2];
                        vert_2_x = (double)vert_grid_in[ind_tri_2];
                        vert_2_y = (double)vert_grid_in[ind_tri_2 + 1];
                        vert_2_z = (double)vert_grid_in[ind_tri_2 + 2];

                        double norm_hori_x, norm_hori_y, norm_hori_z,
                            area_hori;
                        triangle_normal_area(vert_0_x, vert_0_y, vert_0_z,
                            vert_1_x, vert_1_y, vert_1_z,
                            vert_2_x, vert_2_y, vert_2_z,
                            norm_hori_x, norm_hori_y, norm_hori_z,
                        area_hori);

                        double surf_enl_fac = area_tilt / area_hori;

                        //-----------------------------------------------------
                        // Compute horizon in local ENU coordinate system
                        //-----------------------------------------------------

                        // Vector to North Pole (and orthogonal to 'norm_hori')
                        double north_x = (north_pole[0] - cent_x);
                        double north_y = (north_pole[1] - cent_y);
                        double north_z = (north_pole[2] - cent_z);
                        double dot_prod = ((north_x * norm_hori_x)
                                         + (north_y * norm_hori_y)
                                         + (north_z * norm_hori_z));
                        north_x -= dot_prod * norm_hori_x;
                        north_y -= dot_prod * norm_hori_y;
                        north_z -= dot_prod * norm_hori_z;
                        vec_unit(north_x, north_y, north_z);

                        double east_x, east_y, east_z;
                        cross_prod(north_x, north_y, north_z,
                            norm_hori_x, norm_hori_y, norm_hori_z,
                            east_x, east_y, east_z);
                        double rot_inv[3][3] =
                            {{east_x, north_x, norm_hori_x},
                             {east_y, north_y, norm_hori_y},
                             {east_z, north_z, norm_hori_z}};

                        function_pointer(
                            (float)ray_org_x, (float)ray_org_y,
                            (float)ray_org_z,
                            hori_azim_num, hori_acc, dist_search,
                            elev_ang_low_lim, elev_ang_up_lim, elev_num,
                            scene, num_rays, &horizon[0],
                            azim_sin, azim_cos, elev_ang,
                            elev_cos, elev_sin, rot_inv);
                        // values that are no used for operations and are only
                        // passed are already converted to 'float' here

                        //-----------------------------------------------------
                        // Compute sky view factor
                        //-----------------------------------------------------

                        // Rotate tilt vector from global to local ENU
                        // coordinate system
                        double rot[3][3] = {{east_x, east_y, east_z},
                                            {north_x, north_y, north_z},
                                            {norm_hori_x, norm_hori_y,
                                             norm_hori_z}};
                        double tilt_global[3] = {norm_tilt_x, norm_tilt_y,
                                                 norm_tilt_z};
                        double tilt_local[3];
                        mat_vec_mult(rot, tilt_global, tilt_local);
                        tilt_gc[0] = tilt_gc[0] + tilt_local[0];
                        tilt_gc[1] = tilt_gc[1] + tilt_local[1];
                        tilt_gc[2] = tilt_gc[2] + tilt_local[2];

                        // Compute sky view factor
                        double agg = 0.0;
                        for (size_t o = 0; o < hori_azim_num; o++) {

                            // Compute plane-sphere intersection
                            double hori_plane = atan(-azim_sin[o]
                                * tilt_local[0] / tilt_local[2]
                                - azim_cos[o] * tilt_local[1] / tilt_local[2]);
                            double hori_elev;
                            if (horizon[o] >= hori_plane) {
                                hori_elev = horizon[o];
                            } else {
                                hori_elev =  hori_plane;
                            }

                            // Compute inner integral
                            agg = agg + ((tilt_local[0] * azim_sin[o]
                                + tilt_local[1] * azim_cos[o]) * ((M_PI / 2.0)
                                - hori_elev - (sin(2.0 * hori_elev) / 2.0))
                                + tilt_local[2] * pow(cos(hori_elev), 2));

                        }

                        sky_view_factor[lin_ind_gc]
                            = sky_view_factor[lin_ind_gc]
                            + (azim_spac / (2.0 * M_PI)) * agg;

                        area_increase_factor[lin_ind_gc]
                            = area_increase_factor[lin_ind_gc]
                            + surf_enl_fac;

                        sky_view_area_factor[lin_ind_gc]
                            = sky_view_area_factor[lin_ind_gc]
                            + ((azim_spac / (2.0 * M_PI)) * agg)
                            * surf_enl_fac;

                        //-----------------------------------------------------
                        // Loop through sun positions and compute correction
                        // factors
                        //-----------------------------------------------------

                        // Compute sine of horizon
                        for (size_t o = 0; o < hori_azim_num; o++) {
                            horizon_sin[o] = sin(horizon[o]);
                        }

                        // Make horizon data periodical
                        horizon[hori_azim_num] = horizon[0];
                        horizon_sin[hori_azim_num] = horizon_sin[0];

                        // Compute minimal/maximal of sine of horizon
                        double horizon_sin_min = 1.0;
                        double horizon_sin_max = -1.0;
                        for (size_t o = 0; o < hori_azim_num; o++) {
                            if (horizon_sin[o] < horizon_sin_min) {
                                horizon_sin_min = horizon_sin[o];
                            }
                            if (horizon_sin[o] > horizon_sin_max) {
                                horizon_sin_max = horizon_sin[o];
                            }
                        }

                        size_t ind_lin_sun, ind_lin_cor;
                        for (size_t o = 0; o < dim_sun_0; o++) {
                            for (size_t p = 0; p < dim_sun_1; p++) {

                                ind_lin_sun = lin_ind_3d(dim_sun_1, 3,
                                    o, p, 0);

                                // Compute sun unit vector
                                double sun_x = (sun_pos[ind_lin_sun]
                                    - ray_org_x);
                                double sun_y = (sun_pos[ind_lin_sun + 1]
                                    - ray_org_y);
                                double sun_z = (sun_pos[ind_lin_sun + 2]
                                    - ray_org_z);
                                vec_unit(sun_x, sun_y, sun_z);

                                // Check for self-shadowing (Earth)
                                double dot_prod_hs = (norm_hori_x * sun_x
                                    + norm_hori_y * sun_y
                                    + norm_hori_z * sun_z);
                                if (dot_prod_hs <= dot_prod_min) {
                                    continue;  // sw_dir_cor += 0.0
                                }

                                // Check for self-shadowing (triangle)
                                double dot_prod_ts = norm_tilt_x * sun_x
                                    + norm_tilt_y * sun_y
                                    + norm_tilt_z * sun_z;
                                if (dot_prod_ts <= 0.0) {
                                    continue;  // sw_dir_cor += 0.0
                                }

                                // Rotate sun vector from global to local ENU
                                // coordinate system
                                double sun_global[3] = {sun_x, sun_y, sun_z};
                                double sun_local[3];
                                mat_vec_mult(rot, sun_global, sun_local);

                                // Compare sun position to location's overall
                                // minimal/maximal horizon -> separate cases
                                // that require less expensive operations
                                if (sun_local[2] <= horizon_sin_min) {
                                    continue;  // shadow (sw_dir_cor += 0.0)
                                } else if (sun_local[2] <= horizon_sin_max) {
                                    double sun_azim = atan2(sun_local[0],
                                                            sun_local[1]);
                                    if (sun_azim < 0.0) {
                                        sun_azim += (2.0 * M_PI);
                                    }
                                    // range: [0.0 <= 'sun_azim < 2.0 * pi]
                                    int ind_0 = int(sun_azim / azim_spac);
                                    double weight = (sun_azim
                                        - (ind_0 * azim_spac)) / azim_spac;
                                    double horizon_sin_sun = horizon_sin[ind_0]
                                        * (1.0 - weight)
                                        + horizon_sin[ind_0 + 1] * weight;
                                    if (sun_local[2] <= horizon_sin_sun) {
                                        continue;
                                        // shadow (sw_dir_cor += 0.0)
                                    }
                                }

                                // Compute correction factor for illuminated
                                // case
                                ind_lin_cor = lin_ind_4d(num_gc_x,
                                    dim_sun_0, dim_sun_1, i, j, o, p);
                                sw_dir_cor[ind_lin_cor] =
                                    (float)(sw_dir_cor[ind_lin_cor]
                                    + std::min(((dot_prod_ts
                                    / dot_prod_hs) * surf_enl_fac),
                                    sw_dir_cor_max));

                            }
                        }

                    }

                }
            }

            // Compute mean grid cell slope and aspect
            vec_unit(tilt_gc[0], tilt_gc[1], tilt_gc[2]);
            slope[lin_ind_gc] = rad2deg(acos(tilt_gc[2]));
            double aspect_temp = atan2(tilt_gc[0], tilt_gc[1]);
            if (aspect_temp < 0.0) {
                aspect_temp += 2.0 * M_PI;
            }
            aspect[lin_ind_gc] = rad2deg(aspect_temp);

            delete[] horizon;
            delete[] horizon_sin;
            delete[] tilt_gc;

            } else {

                size_t ind_lin = lin_ind_4d(num_gc_x, dim_sun_0, dim_sun_1,
                    i, j, 0, 0);
                for (size_t k = 0; k < (dim_sun_0 * dim_sun_1) ; k++) {
                    sw_dir_cor[ind_lin + k] = NAN;
                }
                sky_view_factor[lin_ind_gc] = NAN;
                area_increase_factor[lin_ind_gc] = NAN;
                sky_view_area_factor[lin_ind_gc] = NAN;
                slope[lin_ind_gc] = NAN;
                aspect[lin_ind_gc] = NAN;

            }

        }
    }

    return num_rays;  // parallel
    }, std::plus<size_t>());  // parallel

    auto end_ray = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_ray = (end_ray - start_ray);
    cout << "Ray tracing time: " << time_ray.count() << " s" << endl;

    // Print number of rays needed for location and azimuth direction
    cout << "Number of rays shot: " << num_rays << endl;
    double gc_proc = 0;
    for (size_t i = 0; i < (num_gc_y * num_gc_x); i++) {
        if (mask[i] == 1) {
            gc_proc += 1;
        }
    }
    double tri_proc = (gc_proc * (double)(pixel_per_gc * pixel_per_gc) * 2.0)
        * (double)hori_azim_num;
    double ratio = (double)num_rays / (double)tri_proc;
    printf("Average number of rays per location and azimuth: %.2f \n", ratio);

    // Divide accumulated values by number of triangles within grid cell
    double num_tri_per_gc = pixel_per_gc * pixel_per_gc * 2.0;
    size_t num_elem = (num_gc_y * num_gc_x);
    for (size_t i = 0; i < num_elem; i++) {
        sky_view_factor[i] /= num_tri_per_gc;
        area_increase_factor[i] /= num_tri_per_gc;
        sky_view_area_factor[i] /= num_tri_per_gc;
    }  
    num_elem = (num_gc_y * num_gc_x * dim_sun_0 * dim_sun_1);
    for (size_t i = 0; i < num_elem; i++) {
        sw_dir_cor[i] /= (float)num_tri_per_gc;
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
// Compute sky view factor and average distance to terrain
//-----------------------------------------------------------------------------

void sky_view_factor_dist_comp(
    float* vert_grid,
    int dem_dim_0, int dem_dim_1,
    float* vert_grid_in,
    int dem_dim_in_0, int dem_dim_in_1,
    double* north_pole,
    double* sky_view_factor,
    double* area_increase_factor,
    double* sky_view_area_factor,
    double* slope,
    double* aspect,
    double* distance,
    int pixel_per_gc,
    int offset_gc,
    uint8_t* mask,
    float dist_search,
    int azim_num,
    int elev_num,
    char* geom_type) {

    cout << "--------------------------------------------------------" << endl;
    cout << "Compute sky view factor " << endl;
    cout << "--------------------------------------------------------" << endl;

    // Hard-coded settings
    double ray_org_elev = 0.1;
    // value to elevate ray origin (-> avoids potential issue with numerical
    // imprecision / truncation) [m]

    // Number of grid cells
    int num_gc_y = (dem_dim_in_0 - 1) / pixel_per_gc;
    int num_gc_x = (dem_dim_in_1 - 1) / pixel_per_gc;
    cout << "Number of grid cells in y-direction: " << num_gc_y << endl;
    cout << "Number of grid cells in x-direction: " << num_gc_x << endl;

    // Number of triangles
    int num_tri = (dem_dim_in_0 - 1) * (dem_dim_in_1 - 1) * 2;
    cout << "Number of triangles: " << num_tri << endl;

    // Unit conversion(s)
    dist_search *= 1000.0;  // [kilometre] to [metre]
    cout << "Search distance: " << dist_search << " m" << endl;

    // Initialisation
    auto start_ini = std::chrono::high_resolution_clock::now();
    RTCDevice device = initializeDevice();
    RTCScene scene = initializeScene(device, vert_grid, dem_dim_0, dem_dim_1,
        geom_type);
    auto end_ini = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = end_ini - start_ini;
    cout << "Total initialisation time: " << time.count() << " s" << endl;

    // ------------------------------------------------------------------------
    // Allocate and initialise arrays with evaluated trigonometric functions
    // ------------------------------------------------------------------------

    // Azimuth angles (allocate on stack)
    double azim_sin[azim_num];
    double azim_cos[azim_num];
    double ang;
    for (int i = 0; i < azim_num; i++) {
        ang = ((2 * M_PI) / azim_num * i);
        azim_sin[i] = sin(ang);
        azim_cos[i] = cos(ang);
    }
    double azim_spac = (2.0 * M_PI) / (double)azim_num;

    // Elevation angles (allocate on stack)
    double beta_step = -2.0 / (double)elev_num;
    double alpha[elev_num + 1];
    for (int i = 0; i < (elev_num + 1); i++) {
        double beta = 1.0 + ((double)i * beta_step);
        alpha[i] = acos(beta) / 2.0;
    }
    double elev_sin[elev_num];
    double elev_cos[elev_num];
    for (int i = 0; i < elev_num; i++) {
        ang = (alpha[i] + alpha[i + 1]) / 2.0;
        elev_sin[i] = sin(ang);
        elev_cos[i] = cos(ang);
    }

    //-------------------------------------------------------------------------

    auto start_ray = std::chrono::high_resolution_clock::now();
    size_t num_rays = 0;

    num_rays += tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, num_gc_y), 0.0,
        [&](tbb::blocked_range<size_t> r, size_t num_rays) {  // parallel

    // Loop through 2D-field of grid cells
    // for (size_t i = 0; i < num_gc_y; i++) {  // serial
    for (size_t i=r.begin(); i<r.end(); ++i) {  // parallel
        for (size_t j = 0; j < num_gc_x; j++) {

            size_t lin_ind_gc = lin_ind_2d(num_gc_x, i, j);
            if (mask[lin_ind_gc] == 1) {

            double* tilt_gc = new double[3] {0.0, 0.0, 0.0};
            double dist_mean_gc = 0.0;
            int num_dist = 0;

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

                        double vert_0_x = (double)vert_grid[ind_tri_0];
                        double vert_0_y = (double)vert_grid[ind_tri_0 + 1];
                        double vert_0_z = (double)vert_grid[ind_tri_0 + 2];
                        double vert_1_x = (double)vert_grid[ind_tri_1];
                        double vert_1_y = (double)vert_grid[ind_tri_1 + 1];
                        double vert_1_z = (double)vert_grid[ind_tri_1 + 2];
                        double vert_2_x = (double)vert_grid[ind_tri_2];
                        double vert_2_y = (double)vert_grid[ind_tri_2 + 1];
                        double vert_2_z = (double)vert_grid[ind_tri_2 + 2];

                        double cent_x, cent_y, cent_z;
                        triangle_centroid(vert_0_x, vert_0_y, vert_0_z,
                            vert_1_x, vert_1_y, vert_1_z,
                            vert_2_x, vert_2_y, vert_2_z,
                            cent_x, cent_y, cent_z);

                        double norm_tilt_x, norm_tilt_y, norm_tilt_z,
                            area_tilt;
                        triangle_normal_area(vert_0_x, vert_0_y, vert_0_z,
                            vert_1_x, vert_1_y, vert_1_z,
                            vert_2_x, vert_2_y, vert_2_z,
                            norm_tilt_x, norm_tilt_y, norm_tilt_z,
                            area_tilt);

                        // Ray origin
                        double ray_org_x = (cent_x
                            + norm_tilt_x * ray_org_elev);
                        double ray_org_y = (cent_y
                            + norm_tilt_y * ray_org_elev);
                        double ray_org_z = (cent_z
                            + norm_tilt_z * ray_org_elev);

                        //-----------------------------------------------------
                        // Horizontal triangle
                        //-----------------------------------------------------

                        func_ptr[n](dem_dim_in_1, k, m,
                            ind_tri_0, ind_tri_1, ind_tri_2);

                        vert_0_x = (double)vert_grid_in[ind_tri_0];
                        vert_0_y = (double)vert_grid_in[ind_tri_0 + 1];
                        vert_0_z = (double)vert_grid_in[ind_tri_0 + 2];
                        vert_1_x = (double)vert_grid_in[ind_tri_1];
                        vert_1_y = (double)vert_grid_in[ind_tri_1 + 1];
                        vert_1_z = (double)vert_grid_in[ind_tri_1 + 2];
                        vert_2_x = (double)vert_grid_in[ind_tri_2];
                        vert_2_y = (double)vert_grid_in[ind_tri_2 + 1];
                        vert_2_z = (double)vert_grid_in[ind_tri_2 + 2];

                        double norm_hori_x, norm_hori_y, norm_hori_z,
                            area_hori;
                        triangle_normal_area(vert_0_x, vert_0_y, vert_0_z,
                            vert_1_x, vert_1_y, vert_1_z,
                            vert_2_x, vert_2_y, vert_2_z,
                            norm_hori_x, norm_hori_y, norm_hori_z,
                        area_hori);

                        double surf_enl_fac = area_tilt / area_hori;

                        //-----------------------------------------------------
                        // Compute surface slope and aspect
                        //-----------------------------------------------------

                        // Vector to North Pole (and orthogonal to 'norm_hori')
                        double north_x = (north_pole[0] - cent_x);
                        double north_y = (north_pole[1] - cent_y);
                        double north_z = (north_pole[2] - cent_z);
                        double dot_prod = ((north_x * norm_hori_x)
                                         + (north_y * norm_hori_y)
                                         + (north_z * norm_hori_z));
                        north_x -= dot_prod * norm_hori_x;
                        north_y -= dot_prod * norm_hori_y;
                        north_z -= dot_prod * norm_hori_z;
                        vec_unit(north_x, north_y, north_z);

                        double east_x, east_y, east_z;
                        cross_prod(north_x, north_y, north_z,
                            norm_hori_x, norm_hori_y, norm_hori_z,
                            east_x, east_y, east_z);

                       // Rotate tilt vector from global to local ENU
                        // coordinate system
                        double rot[3][3] = {{east_x, east_y, east_z},
                                            {north_x, north_y, north_z},
                                            {norm_hori_x, norm_hori_y,
                                             norm_hori_z}};
                        double tilt_global[3] = {norm_tilt_x, norm_tilt_y,
                                                 norm_tilt_z};
                        double tilt_local[3];
                        mat_vec_mult(rot, tilt_global, tilt_local);
                        tilt_gc[0] = tilt_gc[0] + tilt_local[0];
                        tilt_gc[1] = tilt_gc[1] + tilt_local[1];
                        tilt_gc[2] = tilt_gc[2] + tilt_local[2];

                        //-----------------------------------------------------
                        // Compute sky view factor and average distance to
                        // terrain
                        //-----------------------------------------------------

                         // Approximate north vector in titled coordinate
                         // system with global ENU North (-> can be arbitrary)
                         north_x = 0.0;
                         north_y = 1.0;
                         north_z = -norm_tilt_y / norm_tilt_z;
                         vec_unit(north_x, north_y, north_z);
                         cross_prod(north_x, north_y, north_z,
                             norm_tilt_x, norm_tilt_y, norm_tilt_z,
                             east_x, east_y, east_z);
                        double rot_inv[3][3] =
                            {{east_x, north_x, norm_tilt_x},
                             {east_y, north_y, norm_tilt_y},
                             {east_z, north_z, norm_tilt_z}};

                        double dist_mean = 0.0;
                        sample_hemisphere(
                            (float)ray_org_x, (float)ray_org_y,
                            (float)ray_org_z,
                            azim_num, elev_num, dist_search,
                            scene, num_rays, dist_mean,
                            azim_sin, azim_cos,
                            elev_cos, elev_sin, rot_inv);
                        if (!isnan(dist_mean)) {
                            dist_mean_gc = dist_mean_gc + dist_mean;
                            num_dist = num_dist + 1;
                        }

                    }

                }
            }

            // Compute mean grid cell slope and aspect
            vec_unit(tilt_gc[0], tilt_gc[1], tilt_gc[2]);
            slope[lin_ind_gc] = rad2deg(acos(tilt_gc[2]));
            double aspect_temp = atan2(tilt_gc[0], tilt_gc[1]);
            if (aspect_temp < 0.0) {
                aspect_temp += 2.0 * M_PI;
            }
            aspect[lin_ind_gc] = rad2deg(aspect_temp);

            // Mean distance to terrain
            if (dist_mean_gc != 0.0) {
                distance[lin_ind_gc] = dist_mean_gc / (double)num_dist;
            } else {
                distance[lin_ind_gc] = NAN;
            }

            delete[] tilt_gc;

            } else {

                sky_view_factor[lin_ind_gc] = NAN;
                area_increase_factor[lin_ind_gc] = NAN;
                sky_view_area_factor[lin_ind_gc] = NAN;
                slope[lin_ind_gc] = NAN;
                aspect[lin_ind_gc] = NAN;
                distance[lin_ind_gc] = NAN;

            }

        }
    }

    return num_rays;  // parallel
    }, std::plus<size_t>());  // parallel

    auto end_ray = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_ray = (end_ray - start_ray);
    cout << "Ray tracing time: " << time_ray.count() << " s" << endl;

    // Print number of rays needed for location and azimuth direction
    cout << "Number of rays shot: " << num_rays << endl;
    double gc_proc = 0;
    for (size_t i = 0; i < (num_gc_y * num_gc_x); i++) {
        if (mask[i] == 1) {
            gc_proc += 1;
        }
    }
    double tri_proc = (gc_proc * (double)(pixel_per_gc * pixel_per_gc) * 2.0)
        * (double)azim_num;
    double ratio = (double)num_rays / (double)tri_proc;
    printf("Average number of rays per location and azimuth: %.2f \n", ratio);

    // Divide accumulated values by number of triangles within grid cell
    double num_tri_per_gc = pixel_per_gc * pixel_per_gc * 2.0;
    size_t num_elem = (num_gc_y * num_gc_x);
    for (size_t i = 0; i < num_elem; i++) {
        sky_view_factor[i] /= num_tri_per_gc;
        area_increase_factor[i] /= num_tri_per_gc;
        sky_view_area_factor[i] /= num_tri_per_gc;
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
