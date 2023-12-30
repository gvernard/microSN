#include <cmath>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cstring>

#include "cpu_functions.hpp"

/*
SUMMARY:
Function to find the interpolation indices.

INPUT:
\param t -- Time steps already converted to dimensionless units [RE].
\param x -- The sizes of the profiles in units of [RE].
\param Nd -- The size of the data.
\param Nprof -- The number of profiles used in the convolutions.
\param ind -- An array of size Nd that will hold the interpolation indices.

FUNCTION:
Find the left index of the 'x' interval within which the transformed t [in units of RE] fall.
If t falls outside the end of the x array the index will be set to -1.

OUTPUT:
The array of interpolation indices 'ind'.
 */
void find_indices(double* t,double* x,int Nd,int Nprof,int* ind){
  int i = 0;
  double comp = t[0];
  int j = 0;
  while( i<Nd && j<Nprof-1 ){
    if( x[j] <= comp && comp < x[j+1] ){
      ind[i] = j;
      i++;
      comp = t[i];
    } else if( comp > x[Nprof-1] ){
      ind[i] = -1;
      i++;
      comp = t[i];
    } else {
      j++;
    }
  }
  // Check if j=Nprof-1 and report error if so
}



/*
SUMMARY:
Function that sets up an instance of Chi2Vars that contains quantities necessary for calculating chi2.

INPUT:
\param M -- Vector of masses [in units of solar mass].
\param V -- Vector of velocities [in units of km/s].
\param Dfac -- Square root of the fraction of angular diameter distances: DS*DLS/DL [in units of sqrt(Mpc)].
\param Nd -- Size of the data.
\param d -- Data values [dimensionless, it is the ratio of fluxes].
\param t -- Data times [units of days].
\param s -- Uncertainty of the measurements d [dimensionless, d is the ratio of fluxes].
\param Nprof -- Number of profiles for which light curves have been produced, i.e. the number of steps in each simulated light curve.
\param x -- The values of the profile sizes in units of [RE].

FUNCTION:
We start with two arrays: t (Nd) and x (Nprof), describing the time of the data points and the half-light radius of the profiles (in units of RE).
For each image we call 'find_indices' to create the array of indices.
We loop over these arrays to find where both are equal to -1.
This means that the combination of M and V leads to values of t outside the range of our Nprof profiles.
In this case, we assume that there is no microlensing magnification and our model, i.e. mu_A/mu_B, is equal to 1.
There is no need  to perform these calculations on the GPU, which is true only if the t for both images corresponds to x outside our range.
We simply straightforwardly compute the sum of the data minus 1 over the uncertainty for these specific t values only.
This sum is added to the final chi squared at the end of the calculation.
For the indices that are not -1 for both images, we first shorten the arrays accordingly and calculate the interpolation factors.
The latter are fixed for each image.

OUTPUT:
An instance of Chi2Vars with all its variables set.
*/
Chi2Vars setup_chi2_calculation(std::vector<double> M,std::vector<double> V,double Dfac,int Nd,double* d,double* t,double* s,int Nprof,double* x){
  double Rfac = 23.92865*Dfac; // in units of [10^14 cm / sqrt(M_solar)]

  // Transform time from units of [d] to units of the Einstein radius on the source plane [RE]
  // V must be in 10^5 km/s so that the final units of t2x are 1/d
  double t2xA = V[0]*8.64/(Rfac*sqrt(M[0])); // units of [1/d]
  double t2xB = V[0]*8.64/(Rfac*sqrt(M[1])); // units of [1/d]
  double tA[Nd];
  double tB[Nd];
  for(int i=0;i<Nd;i++){
    tA[i] = t2xA*t[i]; // units of RE
    tB[i] = t2xB*t[i]; // units of RE
  }
  
  // Match the Nd time values [s] to the Nprof simulated light curve steps [units of RE]
  int indA[Nd];
  find_indices(tA,x,Nd,Nprof,indA);
  int indB[Nd];
  find_indices(tB,x,Nd,Nprof,indB);

  // Check how many of the data times are within the simulated Nprof range (either image)
  int Njp = Nd;
  for(int i=0;i<Nd;i++){
    if( indA[i] == -1 && indB[i] == -1 ){
      Njp = i;
      break;
    }
  }
  Chi2Vars vars(Njp);
  
  // Shorten arrays from size of Nd to Njp: d, s, indA, indB
  for(int i=0;i<Njp;i++){
    vars.indA[i] = indA[i];
    vars.indB[i] = indB[i];
    vars.new_d[i] = d[i];
    vars.new_s[i] = s[i];
  }
  
  // Calculate the interpolation factors per image
  int a,b;
  for(int i=0;i<Njp;i++){
    a = indA[i];
    if( a != -1 ){
      vars.facA[i] = (tA[i]-x[a])/(x[a+1]-x[a]);
    } else {
      vars.facA[i] = 0;
    }
    b = indB[i];
    if( b != -1 ){
      vars.facB[i] = (tB[i]-x[b])/(x[b+1]-x[b]);
    } else {
      vars.facB[i] = 0;
    }
  }

  return vars;
}


/*
  
  // Calculate integral over z on the GPU

  
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
*/



/*
SUMMARY:
Calculates z and then sorts the precacalculated chi2 values according to it.

INPUT:
\param Nloc -- The size of the sampled pairs.
\param LCA -- A filled instance of SimLC for image A.
\param LCB -- Same for image B.
\param chi2_values -- A pointer to the 'values' variable from a pre-computed instance of Chi2 - this is a pointer to CPU memory.

FUNCTION:
First use the first value of every light curve in LCA and LCB to calculate the ratio, z, of all the possible pairs (size of Nloc*Nloc).
Then, sort z and a sequence of indices together, in order to keep the permutated indices.
Use the permuated indices and a copy of the chi2 values to reorder the chi2_values.

OUTPUT:
The re-arranged values of chi2_values, sorted by z.
*/
void sort_chi2_by_z_CPU(int Nloc,SimLC* LCA,SimLC* LCB,double* chi2_values){
  
  double* z = (double*) malloc(Nloc*Nloc*sizeof(double));
  for(int a=0;a<Nloc;a++){
    for(int b=0;b<Nloc;b++){
      z[a*Nloc+b] = LCA->LC[a*LCA->Nprof]/LCB->LC[b*LCB->Nprof];
    }
  }

      
  std::vector<unsigned int> idx(Nloc*Nloc);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&z](size_t i1, size_t i2) {return z[i1] < z[i2];});
  
  double* tmp = (double*) malloc(Nloc*Nloc*sizeof(double));
  std::memcpy(tmp,chi2_values,Nloc*Nloc*sizeof(double));
  double* tmp_z = (double*) malloc(Nloc*Nloc*sizeof(double));
  std::memcpy(tmp_z,z,Nloc*Nloc*sizeof(double));
  for(int i=0;i<Nloc*Nloc;i++){
    chi2_values[i] = tmp[idx[i]];
    z[i] = tmp_z[idx[i]];
  }

  free(tmp);
  free(tmp_z);
  free(z);
}



/*
SUMMARY:
Calculate the chi2 values on the CPU.

INPUT:
\param chi2_vars -- An instance of Chi2Vars, that containes all the pre-computed variables that facilitate the chi2 calculation.
\param chi2_values -- A pointer to the 'values' variable from an empty instance of Chi2 - this is a pointer to CPU memory.
\param LCA -- A filled instance of SimLC for image A.
\param LCB -- Same for image B.  

FUNCTION:
We loop over the LCA and LCB light curves and form all the possible pairs (size of Nloc*Nloc).
For each combination, we calculate the chi2 value in a third, innermost loop.

OUTPUT:
The computed chi2 values are stored in the provided chi2_values pointer, a pointer to CPU memory from a Chi2 instance.
*/
void calculate_chi2_CPU(Chi2Vars* chi2_vars,double* chi2_values,SimLC* LCA,SimLC* LCB){
  int Nloc  = LCA->Nloc;
  int Nprof = LCA->Nprof;
  double mA,mB,tmp,chi2_val;
  
  for(int a=0;a<Nloc;a++){
    for(int b=0;b<Nloc;b++){

      chi2_val = 0.0;
      for(int i=0;i<chi2_vars->Njp;i++){
	if( chi2_vars->indA[i] == -1 ){
	  mA = 1;
	} else {
	  mA = LCA->LC[a*Nprof+chi2_vars->indA[i]] + chi2_vars->facA[i]*LCA->DLC[a*(Nprof-1)+chi2_vars->indA[i]];
	}
	
	if( chi2_vars->indB[i] == -1 ){
	  mB = 1;
	} else {
	  mB = LCB->LC[b*Nprof+chi2_vars->indB[i]] + chi2_vars->facB[i]*LCB->DLC[b*(Nprof-1)+chi2_vars->indB[i]];
	}
	
	tmp = (chi2_vars->new_d[i] - (mA/mB))/chi2_vars->new_s[i];
	chi2_val += tmp*tmp;
      }
      chi2_values[a*Nloc+b] = chi2_val;

    }
  }
}



/*
SUMMARY:
Bins the computed and sorted chi2 values (size of Nloc*Nloc) into the provided bins (size of Nbins << Nloc*Nloc).

INPUT:
\param binned_chi2 -- A pointer to where the y-values of the bins will be stored for the chi2 (size of Nbins).
\param binned_exp -- A pointer to where the y-values of the bins will be stored for the exp(-0.5*chi2) (size of Nbins).
\param sort_struct -- A filled instance of Chi2SortBins. Here we need the lower_ind, n_per_bin, and Nbins variables.
\param chi2_values -- A pointer to the 'values' variable from a filled instance of Chi2 - this is a pointer to CPU memory.

FUNCTION:
Using the lower_ind and n_per_bin variables we select all the values from the sorted chi2_values that belong into each bin.
We then take their average, and the average of the likelihood, to store as the y-value of each bin.

OUTPUT:
The average of the chi2 and exp(-0.5*chi2) values that fall in each bin in the binned_chi2 and binned_exp arrays.
*/
void bin_chi2_CPU(double* binned_chi2,double* binned_exp,Chi2SortBins* sort_struct,double* chi2_values){
  double sum_exp,sum_chi;
  
  for(int i=0;i<sort_struct->Nbins;i++){
    sum_chi = 0.0;
    sum_exp = 0.0;

    if( sort_struct->n_per_bin[i] > 0 ){      
      for(int k=0;k<sort_struct->n_per_bin[i];k++){
	sum_exp += exp(-0.5*chi2_values[sort_struct->lower_ind[i]+k]);
	sum_chi += chi2_values[sort_struct->lower_ind[i]+k];
      }
      sum_chi /= sort_struct->n_per_bin[i];
      sum_exp /= sort_struct->n_per_bin[i];
    }
    binned_chi2[i] = sum_chi;
    binned_exp[i]  = sum_exp;
  }
}





/*
SUMMARY:

INPUT:
\param M -- Vector of masses [in units of solar mass].

FUNCTION:

OUTPUT:
*/
