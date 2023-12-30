#ifndef GENERIC_GPU_HPP
#define GENERIC_GPU_HPP

#include <vector>
#include <string>

#include "mpd.hpp"
#include "data_structs.hpp"
#include "magnification_map.hpp"

void expanding_source(MagnificationMap* map,std::vector<double> sizes,std::string shape,int Nloc,int* sample_loc_x,int* sample_loc_y,SimLC* LC);
void test_chi2(Chi2* chi2,int offset,int N);
void test_setup_integral(Chi2SortBins* sort_gpu,Chi2SortBins* sort_cpu,int offset,int N);

// Functions with a CPU analogue with the same calling signature
void setup_integral_GPU(SimLC* LCA,SimLC* LCB,Mpd* mpd_ratio,Chi2SortBins* sorted);
void calculate_chi2_GPU(Chi2Vars* chi2_vars,double* chi2_d_values,Chi2SortBins* sorted,SimLC* LCA,SimLC* LCB);
void bin_chi2_GPU(double* binned_chi2,double* binned_exp,Chi2SortBins* sort_struct,double* chi2_d_values);

#endif /* GENERIC_GPU_HPP */
