#ifndef CHI2_FUNCTIONS_HPP
#define CHI2_FUNCTIONS_HPP
#include <vector>

void find_indices(double t2x,double* t,double* x,int Nd,int Nprof,int* ind);
double get_like_single_pair(std::vector<double> M,std::vector<double> V,double Dfac,int Nd,double* d,double* t,double* s,int Nprof,double* x,int Nlocs,double* LCA,double* LCB,double* DLCA,double* DLCB);

double calculate_chi2_GPU(int N,int* indA,int* indB,double* facA,double* facB,double* d,double* s,int Nloc,int Nprof,double* LCA,double* LCB,double* DLCA,double* DLCB);
double calculate_chi2_CPU(int N,int* indA,int* indB,double* facA,double* facB,double* d,double* s,int Nloc,int Nprof,double* LCA,double* LCB,double* DLCA,double* DLCB);
#endif /* CHI2_FUNCTIONS_HPP */
