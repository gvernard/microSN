#include <cstring>
#include <cstdio>

#include "data_structs.hpp"



Chi2Vars::Chi2Vars(int Njp): Njp(Njp) {
  this->indA  = (int*) malloc(this->Njp*sizeof(int));
  this->indB  = (int*) malloc(this->Njp*sizeof(int));
  this->facA  = (double*) malloc(this->Njp*sizeof(double));
  this->facB  = (double*) malloc(this->Njp*sizeof(double));
  this->new_d = (double*) malloc(this->Njp*sizeof(double));
  this->new_s = (double*) malloc(this->Njp*sizeof(double));
}

Chi2Vars::Chi2Vars(const Chi2Vars& other): Chi2Vars(other.Njp){
  std::memcpy(this->indA,other.indA,this->Njp*sizeof(int));
  std::memcpy(this->indB,other.indB,this->Njp*sizeof(int));
  std::memcpy(this->facA,other.facA,this->Njp*sizeof(double));
  std::memcpy(this->facB,other.facB,this->Njp*sizeof(double));
  std::memcpy(this->new_d,other.new_d,this->Njp*sizeof(double));
  std::memcpy(this->new_s,other.new_s,this->Njp*sizeof(double));
}

Chi2Vars::~Chi2Vars(){
  free(indA);
  free(indB);
  free(facA);
  free(facB);
  free(new_d);
  free(new_s);
}



Chi2::Chi2(int Nloc): Nloc(Nloc){
  this->values = (double*) malloc(this->Nloc*this->Nloc*sizeof(double));
  cudaMalloc(&this->d_values,this->Nloc*this->Nloc*sizeof(double));
  printf("Allocated memory: for chi2 of all pairs of light curves, with size Nloc*Nloc <double>: %ld (bytes)\n",Nloc*Nloc*sizeof(double));
}

Chi2::Chi2(const Chi2& other): Chi2(other.Nloc){
  // The copy constructor is not expected to be used
  std::memcpy(this->values,other.values,this->Nloc*this->Nloc*sizeof(double));
  cudaMemcpy(this->d_values,other.d_values,this->Nloc*this->Nloc*sizeof(double),cudaMemcpyDeviceToDevice);
}

Chi2::~Chi2(){
  free(values);
  cudaFree(d_values);
}

void Chi2::transfer_to_CPU(){
  cudaMemcpy(this->values,this->d_values,this->Nloc*this->Nloc*sizeof(double),cudaMemcpyDeviceToHost);
}



SimLC::SimLC(int Nloc,int Nprof): Nloc(Nloc),Nprof(Nprof){
  this->LC  = (double*) malloc(this->Nloc*this->Nprof*sizeof(double));
  this->DLC = (double*) malloc(this->Nloc*(this->Nprof-1)*sizeof(double));
  cudaMalloc(&this->d_LC,this->Nloc*this->Nprof*sizeof(double));
  cudaMalloc(&this->d_DLC,this->Nloc*(this->Nprof-1)*sizeof(double));
  printf("%d %d\n",this->Nprof,this->Nloc);
  printf("Allocated memory: Light curves, (2Nprof-1)*Nloc <double>: %ld (Mb)\n",(2*this->Nprof-1)*this->Nloc*sizeof(double));
};

SimLC::SimLC(const SimLC& other): SimLC(other.Nloc,other.Nprof){
  // The copy constructor is not expected to be used
  std::memcpy(this->LC,other.LC,this->Nloc*this->Nprof*sizeof(double));
  std::memcpy(this->DLC,other.DLC,this->Nloc*(this->Nprof-1)*sizeof(double));
  cudaMemcpy(this->d_LC,other.d_LC,this->Nloc*this->Nprof*sizeof(double),cudaMemcpyDeviceToDevice);
  cudaMemcpy(this->d_DLC,other.d_DLC,this->Nloc*(this->Nprof-1)*sizeof(double),cudaMemcpyDeviceToDevice);
};

SimLC::~SimLC(){
  free(LC);
  free(DLC);
  cudaFree(d_LC);
  cudaFree(d_DLC);
}

void SimLC::transfer_to_CPU(){
  cudaMemcpy(this->LC,this->d_LC,this->Nloc*this->Nprof*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(this->DLC,this->d_DLC,this->Nloc*(this->Nprof-1)*sizeof(double),cudaMemcpyDeviceToHost);
}