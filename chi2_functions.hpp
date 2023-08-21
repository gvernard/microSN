#include <iostream>
#include <cstdlib>
#include <fstream>

void getChi2Cuda(double* like_all,float** f_obs_all,float* df_obs,float** h_mua,float** h_mub,int Nlocs,int Nwave,int Nratios);
void setGridThreads(int& Ngrid,int& Nthreads,int Nwave);

void getChi2CudaCPU(double* like_all,float** f_obs_all,float* df_obs,float** h_mua,float** h_mub,int Nlocs,int Nwave,int Nratios);
