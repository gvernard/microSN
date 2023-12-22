#ifndef GENERIC_CPU_HPP
#define GENERIC_CPU_HPP

#include <vector>
#include "data_structs.hpp"

void find_indices(double* t,double* x,int Nd,int Nprof,int* ind);

Chi2Vars setup_chi2_calculation(std::vector<double> M,std::vector<double> V,double Dfac,int Nd,double* d,double* t,double* s,int Nprof,double* x);

void calculate_chi2_CPU(Chi2Vars* chi2_vars,Chi2* chi2,SimLC* LCA,SimLC* LCB);

#endif /* GENERIC_CPU_HPP */
