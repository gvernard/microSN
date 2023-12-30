#ifndef GENERIC_CPU_HPP
#define GENERIC_CPU_HPP

#include <vector>

#include "mpd.hpp"
#include "data_structs.hpp"

void find_indices(double* t,double* x,int Nd,int Nprof,int* ind);
Chi2Vars setup_chi2_calculation(std::vector<double> M,std::vector<double> V,double Dfac,int Nd,double* d,double* t,double* s,int Nprof,double* x);

// Functions with a GPU analogue with the same calling signature
void setup_integral_CPU(SimLC* LCA,SimLC* LCB,Mpd* mpd_ratio,Chi2SortBins* sorted);
void calculate_chi2_CPU(Chi2Vars* chi2_vars,double* chi2,Chi2SortBins* sorted,SimLC* LCA,SimLC* LCB);
void bin_chi2_CPU(double* binned_chi2,double* binned_exp,Chi2SortBins* sort_struct,double* chi2_values);

#endif /* GENERIC_CPU_HPP */
