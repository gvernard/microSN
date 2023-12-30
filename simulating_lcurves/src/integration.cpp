#include "integration.hpp"

double trapezium(int N,double* f1,double* f2,double* x){
  double integral = 0.0;
  for(int i=1;i<N;i++){
    integral += (f1[i-1]*f2[i-1] + f1[i]*f2[i])*(x[i]-x[i-1])/2.0;
  }
  return integral;
}
