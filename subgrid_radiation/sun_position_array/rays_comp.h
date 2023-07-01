// Copyright (c) 2023 ETH Zurich, Christian R. Steger
// MIT License

#ifndef TESTLIB_H
#define TESTLIB_H

void sw_dir_cor_comp(
    float* vert_grid,
    int dem_dim_0, int dem_dim_1,
    float* vert_grid_in,
    int dem_dim_in_0, int dem_dim_in_1,
    double* sun_pos,
    int dim_sun_0, int dim_sun_1,
    float* sw_dir_cor,
    int pixel_per_gc,
    int offset_gc,
    uint8_t* mask,
    double dist_search,
    char* geom_type,
    double sw_dir_cor_max,
    double ang_max);

void sw_dir_cor_comp_coherent(
    float* vert_grid,
    int dem_dim_0, int dem_dim_1,
    float* vert_grid_in,
    int dem_dim_in_0, int dem_dim_in_1,
    double* sun_pos,
    int dim_sun_0, int dim_sun_1,
    float* sw_dir_cor,
    int pixel_per_gc,
    int offset_gc,
    uint8_t* mask,
    double dist_search,
    char* geom_type,
    double sw_dir_cor_max,
    double ang_max);

void sw_dir_cor_comp_coherent_rp8(
    float* vert_grid,
    int dem_dim_0, int dem_dim_1,
    float* vert_grid_in,
    int dem_dim_in_0, int dem_dim_in_1,
    double* sun_pos,
    int dim_sun_0, int dim_sun_1,
    float* sw_dir_cor,
    int pixel_per_gc,
    int offset_gc,
    uint8_t* mask,
    double dist_search,
    char* geom_type,
    double sw_dir_cor_max,
    double ang_max);

#endif
