#include <cmath>

#include "chi2_functions.hpp"

__constant__ double d_d[50];
__constant__ double d_s[50];
__constant__ double d_facA[50];
__constant__ double d_facB[50];
__constant__ int d_indA[50];
__constant__ int d_indB[50];

__global__ void kernelChi2(int loop_ind,int N,int Nprof,double* LCA,double* LCB,int Nloc,double* chi2_all,double* z_all);


__global__ void kernelChi2(int loop_ind,int N,int Nprof,double* LCA,double* LCB,int Nloc,double* chi2_all,double* z_all){
  int Nt  = blockDim.x; // equal to 1024
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int a = bid*Nt + tid;
  double chi2,mB,tmp;
  int b;

  // Get all the values of mA just once for each thread
  double mA[50];
  for(int i=0;i<N;i++){
    // Interpolate magnification A
    if( d_indA[i] == -1 ){
      mA[i] = 1;
    } else {
      mA[i] = LCA[a*Nprof+d_indA[i]] + d_facA[i]*( LCA[a*Nprof+d_indA[i]+1] - LCA[a*Nprof+d_indA[i]] );
    }
  }
  __syncthreads();


  for(int j=0;j<Nt;j++){
    b = ((bid+loop_ind)%gridDim.x)*Nt + (tid+j)%Nt;

    chi2 = 0.0;
    for(int i=0;i<N;i++){
      // Interpolate magnification B
      if( d_indB[i] == -1 ){
	mB = 1;
      } else {
	mB = LCB[b*Nprof+d_indB[i]] + d_facB[i]*( LCB[b*Nprof+d_indB[i]+1] - LCB[b*Nprof+d_indB[i]] );
      }
      
      // Calculate chi2 term
      tmp = (d_d[i] - (mA[i]/mB))/d_s[i];
      chi2 += tmp*tmp;
    }
    chi2_all[a*Nloc+b] = chi2;

    z_all[a*Nloc+b] = LCA[a*Nprof]/LCB[b*Nprof];
    
    __syncthreads();
  }

}


double calculate_chi2_GPU(int N,int* indA,int* indB,double* facA,double* facB,double* d,double* s,int Nloc,int Nprof,double* LCA,double* LCB){  
  // Transfer the fixed arrays to the GPU
  cudaMemcpyToSymbol(d_d,d,N*sizeof(double),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_s,s,N*sizeof(double),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_facA,facA,N*sizeof(double),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_facB,facB,N*sizeof(double),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_indA,indA,N*sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_indB,indB,N*sizeof(int),0,cudaMemcpyHostToDevice);
    
  // Transfer LC for both images to the GPU
  double* d_LCA;
  cudaMalloc(&d_LCA,Nloc*Nprof*sizeof(double));
  cudaMemcpy(d_LCA,LCA,Nloc*Nprof*sizeof(double),cudaMemcpyHostToDevice);
  double* d_LCB;
  cudaMalloc(&d_LCB,Nloc*Nprof*sizeof(double));
  cudaMemcpy(d_LCB,LCB,Nloc*Nprof*sizeof(double),cudaMemcpyHostToDevice);

  // Allocate memory for chi2 and z values
  double* d_chi2;
  cudaMalloc(&d_chi2,Nloc*Nloc*sizeof(double));
  double* d_z;
  cudaMalloc(&d_z,Nloc*Nloc*sizeof(double));
  
  // Call chi2 kernel
  int Nblocks = (int) ceil(Nloc/1024);
  for(int k=0;k<Nblocks;k++){
    kernelChi2<<<Nblocks,1024>>>(k,N,Nprof,d_LCA,d_LCB,Nloc,d_chi2,d_z);
  }
  cudaFree(d_LCA);
  cudaFree(d_LCB);



  // Bin exp(-1/2 chi2) by z

  // Loop over z bins and multiply the binned chi2 with the z prior value 

  // Loop over z bins and perform the integration
  double integral = 1.0;
  

  cudaFree(d_chi2);
  cudaFree(d_z);  
  return log(integral);
}