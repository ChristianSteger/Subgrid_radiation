// Copyright (c) 2023 ETH Zurich, Christian R. Steger
// MIT License

#include "sun_position_comp.h"
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
#include <sstream>
#include <iomanip>

using namespace std;
using namespace shapes;

//#############################################################################
// Auxiliary functions
//#############################################################################

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
	   ang: angle [degree]*/
	return ((ang / M_PI) * 180.0);
}

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
// Initialise terrain
//#############################################################################

CppTerrain::CppTerrain() {
    
    device = initializeDevice();
    
}

CppTerrain::~CppTerrain() {

  	// Release resources allocated through Embree
  	rtcReleaseScene(scene);
  	rtcReleaseDevice(device);

}

void CppTerrain::initialise(float* vert_grid,
	int dem_dim_0, int dem_dim_1,
	int offset_0, int offset_1,
	float* vec_tilt,
	float* vec_norm,
	int dim_in_0, int dim_in_1,
	float* surf_enl_fac,
	char* geom_type,
	float ang_max) {

	dem_dim_0_cl = dem_dim_0;
	dem_dim_1_cl = dem_dim_1;
	vert_grid_cl = vert_grid;
	offset_0_cl = offset_0;
	offset_1_cl = offset_1;	
	vec_tilt_cl = vec_tilt;
	vec_norm_cl = vec_norm;
	dim_in_0_cl = dim_in_0;
	dim_in_1_cl = dim_in_1;
	surf_enl_fac_cl = surf_enl_fac;
	ang_max_cl = ang_max;

	auto start_ini = std::chrono::high_resolution_clock::now();

	scene = initializeScene(device, vert_grid, dem_dim_0, dem_dim_1,
		geom_type);
	
	auto end_ini = std::chrono::high_resolution_clock::now();
  	std::chrono::duration<double> time = end_ini - start_ini;
  	cout << "Total initialisation time: " << time.count() << " s" << endl;

}

//#############################################################################
// Compute subgrid correction factor for direct downward shortwave radiation
//#############################################################################

void CppTerrain::sw_dir_cor(float* sun_position, float* sw_dir_cor_buffer) {

	float ray_org_elev=0.05;
	float dot_prod_min = cos(deg2rad(ang_max_cl));

	tbb::parallel_for(tbb::blocked_range<size_t>(0,dim_in_0_cl),
		[&](tbb::blocked_range<size_t> r) {  // parallel

	//for (size_t i = 0; i < (size_t)dim_in_0_cl; i++) {  // serial
	for (size_t i=r.begin(); i<r.end(); ++i) {  // parallel
  		for (size_t j = 0; j < (size_t)dim_in_1_cl; j++) {
  		
  			size_t ind_arr = lin_ind_2d(dim_in_1_cl, i, j);

    			// Get components of terrain surface / ellipsoid normal vectors
    			size_t ind_vec = lin_ind_2d(dim_in_1_cl, i, j) * 3;
  				float tilt_x = vec_tilt_cl[ind_vec];
  				float norm_x = vec_norm_cl[ind_vec];
  				ind_vec += 1;
  				float tilt_y = vec_tilt_cl[ind_vec];
  				float norm_y = vec_norm_cl[ind_vec];
  				ind_vec += 1;
  				float tilt_z = vec_tilt_cl[ind_vec];
  				float norm_z = vec_norm_cl[ind_vec];
  
  				// Ray origin
  				size_t ind_2d = lin_ind_2d(dem_dim_1_cl, i + offset_0_cl,
  					j + offset_1_cl);
  				float ray_org_x = (vert_grid_cl[ind_2d * 3 + 0] 
  					+ norm_x * ray_org_elev);
  				float ray_org_y = (vert_grid_cl[ind_2d * 3 + 1] 
  					+ norm_y * ray_org_elev);
  				float ray_org_z = (vert_grid_cl[ind_2d * 3 + 2] 
  					+ norm_z * ray_org_elev);

  				// Compute sun unit vector
  				float sun_x = (sun_position[0] - ray_org_x);
  				float sun_y = (sun_position[1] - ray_org_y);
  				float sun_z = (sun_position[2] - ray_org_z);
  				vec_unit(sun_x, sun_y, sun_z);

  				float dot_prod_ns = (norm_x * sun_x + norm_y * sun_y
  					+ norm_z * sun_z);
  			
  				// Check for self-shadowing
  				float dot_prod_ts = tilt_x * sun_x + tilt_y * sun_y 
  					+ tilt_z * sun_z;
  				if (dot_prod_ts > dot_prod_min) {
  			
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
  					ray.tfar = std::numeric_limits<float>::infinity();

  					// Intersect ray with scene
  					rtcOccluded1(scene, &context, &ray);

					if (ray.tfar < 0.0) {
						sw_dir_cor_buffer[ind_arr] = 0.0;
					} else {
						if (dot_prod_ns < dot_prod_min) {
							dot_prod_ns = dot_prod_min;
						}
						sw_dir_cor_buffer[ind_arr] = ((dot_prod_ts 
							/ dot_prod_ns) * surf_enl_fac_cl[ind_arr]);	
					}
			
				} else {
			
					sw_dir_cor_buffer[ind_arr] = 0.0;
			
				}
	
		}
	}
	
	}); // parallel

}


