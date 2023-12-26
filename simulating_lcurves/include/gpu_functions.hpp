#ifndef GENERIC_GPU_HPP
#define GENERIC_GPU_HPP

#include <vector>
#include <string>

#include "mpd.hpp"
#include "magnification_map.hpp"
#include "data_structs.hpp"

void expanding_source(MagnificationMap* map,std::vector<double> sizes,std::string shape,int Nloc,int* sample_loc_x,int* sample_loc_y,SimLC* LC);

void setup_integral(SimLC* LCA,SimLC* LCB,Mpd* mpd_ratio,Chi2SortBins* sorted);

void calculate_chi2_GPU(Chi2Vars* chi2_vars,Chi2* chi2,Chi2SortBins* sorted,SimLC* LCA,SimLC* LCB);

void sort_chi2_by_z_GPU(int Nloc,Chi2SortBins* sort_struct,Chi2* chi2);

void test_chi2(Chi2* chi2,int offset,int N,int Nloc);

void bin_chi2_GPU(int Nloc,double* binned_chi2,double* binned_exp,Chi2SortBins* sort_struct,Chi2* chi2);

#endif /* GENERIC_GPU_HPP */
