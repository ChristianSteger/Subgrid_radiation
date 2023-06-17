#include <embree3/rtcore.h>

namespace shapes {
class CppTerrain {
public:
    RTCDevice device;
    RTCScene scene;
    int dem_dim_0_cl, dem_dim_1_cl;
    float* vert_grid_cl;
    int dem_dim_in_0_cl, dem_dim_in_1_cl;
    float* vert_grid_in_cl;
    int pixel_per_gc_cl;
    int offset_gc_cl;
    float dist_search_cl;
    float ang_max_cl;
    float sw_dir_cor_max_cl;
    float ray_org_elev_cl;
    float dot_prod_rem_cl;
    int num_gc_y_cl, num_gc_x_cl;
    int num_tri_cl;
    float dot_prod_min_cl;
    CppTerrain();
    ~CppTerrain();
    void initialise(
    	float* vert_grid,
    	int dem_dim_0, int dem_dim_1,
		float* vert_grid_in,
		int dem_dim_in_0, int dem_dim_in_1,
		int pixel_per_gc,
		int offset_gc,
		float dist_search,
		char* geom_type,
		float ang_max,
		float sw_dir_cor_max);
    void sw_dir_cor(float* sun_pos, float* sw_dir_cor);
    void sw_dir_cor_coherent(float* sun_pos, float* sw_dir_cor);
    void sw_dir_cor_coherent_rp8(float* sun_pos, float* sw_dir_cor);
};
}
