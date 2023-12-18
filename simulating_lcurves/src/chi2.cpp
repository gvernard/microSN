#include <vector>
#include <cmath>

#include "chi2_functions.hpp"

/*


*/
  // Calculate partial differences within each LCA and LCB

  // Calculate z for the combination of examples between LCA and LCB


void find_indices(double t2x,double* t,double* x,int Nd,int Nprof,int* ind){
  int i = 0;
  double comp = t2x*t[0];
  int j = 0;
  while( i<Nd && j<Nprof-1 ){
    if( x[j] <= comp && comp < x[j+1] ){
      ind[i] = j;
      i++;
      comp = t2x*t[i];
    } else if( comp > x[Nprof-1] ){
      ind[i] = -1;
      i++;
      comp = t2x*t[i];
    } else {
      j++;
    }
  }
  // Check if j=Nprof-1 and report error if so
  
}



/*******************************************************************************
Kernel to calculate the chi-squared likelihood on the GPU.

We start with two arrays: t (Nd) and x (Nprof), describing the time of the data
points and the half-light radius of the profiles (in units of RE).

For each image we create the array of indices:
ind (Nd): The array of the left index of the x interval between which t falls.
          If t falls outside the end of the x array the index will be -1.

We loop over these arrays to find where both are -1.
This means that the combination of M and V leads to values t outside the range
of our Nprof profiles. In this case, we assume that there is no microlensing
magnification and our model, i.e. mu_A/mu_B, is equal to 1. There is no need 
to perform these calculations on the GPU (true only if the t for both images
corresponds to x outside our range), so we just add a straightforwardly computed 
term, which is the sum of the data minus 1 over the uncertainty for these
specific t values only, to the final chi squared at the end of the calculation.

For the indices that are not -1 for both images we proceed with calling the
GPU kernel, but first we shorten the arrays accordingly and calculate the 
interpolation factors. The latter are fixed for each image.



INPUT
\param M -- Vector of masses [in units of solar mass].
\param V -- Vector of velocities [in units of km/s].
\param Dfac -- Square root of the fraction of angular diameter distances:
               DL*DLS/DS [in units of sqrt(Mpc)].
\param Nd -- Size of the data.
\param d -- Data values [dimensionless, it is the ratio of fluxes].
\param t -- Data times [units of s].
\param s -- Uncertainty of the measurements d
            [dimensionless, d is the ratio of fluxes].
\param Nprof -- Number of profiles for which light curves have been produced.
                i.e. the number of steps in each simulated light curve.
\param x -- The values of the profile sizes in units of [RE]
\param Nloc -- Number of simulated light curves per magnification map.
\param LCA -- The location in memory where the Nloc light curves
              of length Nprof are stored (for one image).
\param LCB -- Same as LCA.
*******************************************************************************/
double get_like_single_pair(std::vector<double> M,std::vector<double> V,double Dfac,int Nd,double* d,double* t,double* s,int Nprof,double* x,int Nloc,double* LCA,double* LCB){
  double Rfac = 23.92865*Dfac; // in units of [10^14 cm / sqrt(M_solar)]

  // Factors to transform time in units of [s] to units of the Einstein radius on the source plane [RE]
  double t2xA = V[0]/(Rfac*sqrt(M[0]));
  double t2xB = V[0]/(Rfac*sqrt(M[1]));
  
  // Match the Nd time values [s] to the Nprof simulated light curve steps [units of RE]
  int indA[Nd];
  find_indices(t2xA,t,x,Nd,Nprof,indA);
  int indB[Nd];
  find_indices(t2xB,t,x,Nd,Nprof,indB);

  // Check how many of the data times are within the simulated Nprof range (either image)
  int Njp = 0;
  for(int i=0;i<Nd;i++){
    if( indA[i] == -1 && indB[i] == -1 ){
      Njp = i;
      break;
    }
  }
  
  // Shorten arrays from size of Nd to Njp: d, s, indA, indB
  int new_indA[Njp];
  int new_indB[Njp];
  double new_d[Njp];
  double new_s[Njp];
  for(int i=0;i<Njp;i++){
    new_indA[i] = indA[i];
    new_indB[i] = indB[i];
    new_d[i] = d[i];
    new_s[i] = s[i];
  }
  
  // Calculate the interpolation factors per image
  int a,b;
  double facA[Njp];
  double facB[Njp];
  for(int i=0;i<Njp;i++){
    a = indA[i];
    if( a != -1 ){
      facA[i] = (t2xA*t[i]-x[a])/(x[a+1]-x[a]);
    } else {
      facA[i] = 0;
    }
    b = indB[i];
    if( b != -1 ){
      facB[i] = (t2xB*t[i]-x[b])/(x[b+1]-x[b]);
    } else {
      facB[i] = 0;
    }
  }

  // Calculate integral over z on the GPU
  double log_integral = calculate_chi2_GPU(Njp,indA,indB,facA,facB,d,s,Nloc,Nprof,LCA,LCB);
  
  // Add constant term to the integral
  if( Njp != Nd ){
    double extra_sum = 0.0;
    for(int i=Nd-Njp;i<Nd;i++){
      extra_sum += pow((d[i]-1)/s[i],2);
    }
    log_integral += -extra_sum/2.0;
  }

  return log_integral;
}









