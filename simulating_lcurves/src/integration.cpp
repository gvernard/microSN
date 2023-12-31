#include "integration.hpp"

double trapezium(int N,double* f1,double* f2,double* x){
  double integral = 0.0;
  for(int i=1;i<N;i++){
    integral += (f1[i-1]*f2[i-1] + f1[i]*f2[i])*(x[i]-x[i-1])/2.0;
  }
  return integral;
}


double simpson(int N,double* f1,double* f2,double* x){
  double integral = 0.0;
  double denA,denB,denC,numA,numB,numC;
  double gA,gB,gC,gP;
  double xp;

  for(int i=2;i<N;i=i+2){
    xp = (x[i]+x[i-2])/2.0;

    numA = (xp - x[i-1])*(xp - x[i]);
    numB = (xp - x[i-2])*(xp - x[i]);
    numC = (xp - x[i-2])*(xp - x[i-1]);
    
    denA = (x[i-2] - x[i-1])*(x[i-2]-x[i]);
    denB = (x[i-1] - x[i-2])*(x[i-1]-x[i]);
    denC = (x[i] - x[i-2])*(x[i]-x[i-1]);
    
    gA = f1[i-2]*f2[i-2];
    gB = f1[i-1]*f2[i-1];
    gC = f1[i]*f2[i];
    gP = gA*numA/denA + gB*numB/denB + gC*numC/denC;

    integral += (gA + 4*gP + gC)*(x[i]-x[i-2])/6.0;
  }

  if( N%2 != 0 ){
    integral += (f1[N-2]*f2[N-2] + f1[N-1]*f2[N-1])*(x[N-1]-x[N-2])/2.0;
  }
  
  return integral;
}
