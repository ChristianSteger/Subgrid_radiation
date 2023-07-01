// Copyright (c) 2023 ETH Zurich, Christian R. Steger
// MIT License

#ifndef TESTLIB_H
#define TESTLIB_H

void sky_view_factor_comp(
    float* vert_grid,
    int dem_dim_0, int dem_dim_1,
    float* vert_grid_in,
    int dem_dim_in_0, int dem_dim_in_1,
    double* sky_view_factor,
    double* area_increase_factor,
    double* sky_view_area_factor,
    int pixel_per_gc,
    int offset_gc,
    uint8_t* mask,
    float dist_search,
    int hori_azim_num,
    double hori_acc,
    char* ray_algorithm,
    double elev_ang_low_lim,
    char* geom_type);

void sky_view_factor_sw_dir_cor_comp(
    float* vert_grid,
    int dem_dim_0, int dem_dim_1,
    float* vert_grid_in,
    int dem_dim_in_0, int dem_dim_in_1,
    double* sun_pos,
    int dim_sun_0, int dim_sun_1,
    float* sw_dir_cor,
    double* sky_view_factor,
    double* area_increase_factor,
    double* sky_view_area_factor,
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
    double ang_max);

#endif
