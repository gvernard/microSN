#include <cmath>
#include <cstdio>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <cufft.h>

#include "data_structs.hpp"
#include "gpu_functions.hpp"
#include "magnification_map.hpp"
#include "profile.hpp"
#include "util.hpp"

__constant__ double d_d[50];
__constant__ double d_s[50];
__constant__ double d_facA[50];
__constant__ double d_facB[50];
__constant__ int d_indA[50];
__constant__ int d_indB[50];

__constant__ unsigned int d_lower_ind[5120];
__constant__ unsigned int d_n_per_bin[5120];

__global__ void kernel_chi2(int loop_ind,int N,int Nprof,double* d_LCA,double* d_LCB,double* d_DLCA,double* d_DLCB,int Nloc,double* chi2_d_values,unsigned int* sorted);
__global__ void kernel_z(int k,int Nloc,int Nprof,double* d_LCA,double* d_LCB,double* d_z);
__global__ void kernel_multiplyFFTs(cufftDoubleComplex* Fmap,cufftDoubleComplex* Fkernel,double norm);
__global__ void kernel_sample(cufftDoubleReal* convmap,double* d_LC,double* d_DLC,int k,int Nprof,int* loc_x,int* loc_y,int Nx,int Nloc);
__global__ void kernel_bin_chi2(int Nbins,double* d_binned_chi2,double* d_binned_exp,double* d_chi2);
void myfft2d_r2c(cufftHandle* plan, cufftDoubleReal* data, cufftDoubleComplex* Fdata);
void myfft2d_c2r(cufftHandle* plan, cufftDoubleComplex* Fdata, cufftDoubleReal* data);


/*
SUMMARY:
Calculates light curves of an expanding source.

INPUT:
\param map -- A filled MagnificationMap instance.
\param sizes -- A vector<double> of the profile half-light radii in units of RE (length of Nprof). The first entry must always be equal to 0.
\param shape -- The shape of the profile.
\param Nloc -- The number of sampled locations.
\param sample_loc_x -- An array of int of size Nloc that contains the x index of the sampled pixels.
\param sample_loc_y -- An array of int of size Nloc that contains the y index of the sampled pixels.
\param LC -- A pointer to an empty SimLC object of size Nloc, whose d_LC and d_DLC arrays (pointers in GPU memory) will be filled by this function.

FUNCTION:
First we initialize a few variables, most importantly the size of the grids and blocks of the CUDA kernels and the light profile of the source.
Then, some memory needs to be allocated on the GPU:
1. To store the Fourier transform of the magnification map.
2. To store the Fourier transform of the convolution kernel (changes per loop iteration, see below).
3. To store the convolved map. This memory is also used to read-in the map and kernel before transforming them.
4. There is extra memory needed to run the CUFFT plans.
We proceed with the main operation of this function by transferring the map to the GPU, sampling it, and taking its Fourier transform.
Inside a loop over the profile sizes that have half-light radii >0 we:
- create the profile according to the given shape
- create the corredponding kernel at the predefined CPU memory location
- get the Fourier transform of the kernel
- multiply it with the Fourier transform of the map
- get inverse Fourier transform of the product
- sample the convolved map at the given pixels and store them in the LC object

OUTPUT:
The LC instance of SimLC contains two GPU memory pointers, LC and DLC, that are filled by this function.
These are the light curve values (size of Nloc*Nprof) and the differences between adjacent profiles for each light curve (size of Nloc*(Nprof-1)).

NOTES: 
The first profile is ignored and its half-light radius should always be 0.
This is because the first entry in the light curves should be the magnification of a point source, i.e. sampled from the unconvolved map.
Convolving with a profile with a half-light radius equal to 0 will have the same result, but a convolution would be wasted.
*/
void expanding_source(MagnificationMap* map,std::vector<double> sizes,std::string shape,int Nloc,int* sample_loc_x,int* sample_loc_y,SimLC* LC){

  // ############################################## Initialization ############################################################
  int Nprof = sizes.size();

  // Important definitions for the grids of blocks and blocks of threads
  dim3 block_mult(1000); // Nx/f
  dim3 grid_mult(10,5001); // f,Ny/2+1
  dim3 block_samp(1024);
  dim3 grid_samp((int) ceil((double) Nloc/1024.0));
  
  int Nx = map->Nx;
  int Ny = map->Ny;
  double norm = Nx*Ny;
  //map.writeImageFITS("map.fits",10);
  //MagnificationMap dum_map = map; // A test map on the CPU
  
  // Calculate the maximum offset in pixels
  double max_size = sizes.back(); // in units of Rein
  int maxOffset = (int) ceil( (Nx/map->width)*max_size );
  Kernel kernel(Nx,Ny);

  // Create profile parameters
  std::map<std::string,std::string> profile_pars;
  profile_pars.insert(std::pair<std::string,std::string>("shape", shape));
  profile_pars.insert(std::pair<std::string,std::string>("rhalf", "dum"));
  profile_pars.insert(std::pair<std::string,std::string>("pixSizePhys", std::to_string(map->pixSizePhys)));
  profile_pars.insert(std::pair<std::string,std::string>("incl", "0"));
  profile_pars.insert(std::pair<std::string,std::string>("orient", "0"));
  // ##########################################################################################################################

  
  
  // ############################################## Memory Allocation on the GPU ##############################################
  printf("Used GPU memory:  %d (Mb)\n",(int) get_used_gpu_mem());

  
  // Allocate memory: for the sampled locations
  int* loc_x;
  cudaMalloc((void**) &loc_x,Nloc*sizeof(int));
  cudaMemcpy(loc_x,sample_loc_x,Nloc*sizeof(int),cudaMemcpyHostToDevice);
  int* loc_y;
  cudaMalloc((void**) &loc_y,Nloc*sizeof(int));
  cudaMemcpy(loc_y,sample_loc_y,Nloc*sizeof(int),cudaMemcpyHostToDevice);
  printf("Allocated memory: positions x and y, 2xNloc <int>: %d (Mb)\n",2*Nloc*4);

  // Allocate memory: for the Fourier transform of the map on the GPU (COMPLEX)
  cufftDoubleComplex* Fmap_GPU;
  cudaMalloc( (void**) &Fmap_GPU, Nx*(Ny/2+1)*sizeof(cufftDoubleComplex));
  if (cuda_error("Cuda error: Failed to allocate Fmap_GPU", false, __FILE__, __LINE__)) throw std::bad_alloc();
  printf("Allocated memory: Fourier transform of map, Nx(Ny/2+1) <double complex>: %d (Mb)\n",Nx*(Ny/2+1)*16);

  // Allocate memory: for the Fourier transform of the kernel on the GPU (COMPLEX)
  cufftDoubleComplex* Fkernel_GPU;
  cudaMalloc( (void**) &Fkernel_GPU, Nx*(Ny/2+1)*sizeof(cufftDoubleComplex));
  if (cuda_error("Cuda error: Failed to allocate Fkernel_GPU", false, __FILE__, __LINE__)) throw std::bad_alloc();
  printf("Allocated memory: Fourier transform of kernel, Nx(Ny/2+1) <double complex>: %d (Mb)\n",Nx*(Ny/2+1)*16);

  // Allocate memory: to be used for both the map and the convolved map on the GPU (REAL)
  cufftDoubleReal* any_map_GPU;
  cudaMalloc((void**) &any_map_GPU, Nx*Ny*sizeof(cufftDoubleReal));
  if (cuda_error("Cuda error: Failed to allocate any_map_GPU", false, __FILE__, __LINE__)) throw std::bad_alloc();
  printf("Allocated memory: Any real map, Nx*Ny <double>: %d (Mb)\n",Nx*Ny*8);

  // Create CUFFT plans
  cufftResult result;
  size_t plan_size;

  cufftHandle plan_r2c;
  result = cufftPlan2d(&plan_r2c,Nx,Ny,CUFFT_D2Z);
  if (cuda_error("Cuda error: Failed to create plan", false, __FILE__, __LINE__)) throw std::bad_alloc();
  result = cufftGetSize2d(plan_r2c,Nx,Ny,CUFFT_D2Z,&plan_size);
  printf("Allocated memory: plan R2C: %d (Mb)\n",(int) plan_size);

  cufftHandle plan_c2r;
  result = cufftPlan2d(&plan_c2r,Nx,Ny,CUFFT_Z2D);
  if (cuda_error("Cuda error: Failed to create plan", false, __FILE__, __LINE__)) throw std::bad_alloc();  
  result = cufftGetSize2d(plan_c2r,Nx,Ny,CUFFT_Z2D,&plan_size);
  printf("Allocated memory: plan C2R: %d (Mb)\n",(int) plan_size);


  printf("Used GPU memory:  %d (Mb)\n",(int) get_used_gpu_mem());
  cudaDeviceSynchronize();
  // ##########################################################################################################################



  // ############################################## Operations of the GPU #####################################################
  // Do the Fourier transform of the emap and store it on the GPU
  cudaMemcpy( any_map_GPU, map->data, Nx*Ny*sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  kernel_sample<<<grid_samp,block_samp>>>(any_map_GPU,LC->d_LC,LC->d_DLC,0,Nprof,loc_x,loc_y,Nx,Nloc);
  cudaDeviceSynchronize();
  myfft2d_r2c(&plan_r2c,any_map_GPU,Fmap_GPU);
  cudaDeviceSynchronize();


  // Loop over the kernels
  for(int k=1;k<Nprof;k++){
    std::cout << "Profile: " << k << std::endl;

    // Create profile and kernel
    profile_pars["rhalf"] = std::to_string(sizes[k]*map->pixSizePhys*Nx/map->width);
    BaseProfile* profile = FactoryProfile::getInstance()->createProfileFromHalfRadius(profile_pars);
    kernel.setKernel(profile);
    delete(profile);

    // Fourier transform of kernel
    cudaMemcpy( any_map_GPU, kernel.data, Nx*Ny*sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    myfft2d_r2c(&plan_r2c,any_map_GPU,Fkernel_GPU);
    cudaDeviceSynchronize();
  
    // Multiply Fourier transforms of map and kernel
    kernel_multiplyFFTs<<<grid_mult,block_mult>>>(Fmap_GPU,Fkernel_GPU,norm);
    cudaDeviceSynchronize();

    // Get inverse Fourier transform of product
    myfft2d_c2r(&plan_c2r,Fkernel_GPU,any_map_GPU);
    cudaDeviceSynchronize();

    // Transfer convolved map to CPU and write image.
    //cudaMemcpy(dum_map.data,convmap_GPU,Nx*Ny*sizeof(double),cudaMemcpyDeviceToHost);
    //dum_map.writeImageFITS("conv_"+std::to_string(k)+".fits",10);

    // Sample convolved map
    kernel_sample<<<grid_samp,block_samp>>>(any_map_GPU,LC->d_LC,LC->d_DLC,k,Nprof,loc_x,loc_y,Nx,Nloc);
    cudaDeviceSynchronize();
  }
  // ##########################################################################################################################



  
  // ############################################## Cleanup ###################################################################
  cudaFree(Fkernel_GPU);
  cudaFree(Fmap_GPU);
  cudaFree(any_map_GPU);
  cudaFree(loc_x);
  cudaFree(loc_y);
  cufftDestroy(plan_r2c);
  cufftDestroy(plan_c2r);
  // ##########################################################################################################################
}



/*
SUMMARY:

INPUT:
\param M -- Vector of masses [in units of solar mass].

FUNCTION:

OUTPUT:
*/
void test_chi2(Chi2* chi2,int offset,int N){
  int Nloc = chi2->Nloc;
  
  // Fetch GPU chi2 terms
  double* test_chi2 = (double*) malloc(Nloc*Nloc*sizeof(double));
  cudaMemcpy(test_chi2,chi2->d_values,Nloc*Nloc*sizeof(double),cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Print GPU and CPU chi2 terms and their difference
  printf("%10s %10s %10s %10s\n","index","GPU","CPU","diff");
  for(int i=0;i<N;i++){
    double diff = test_chi2[offset+i] - chi2->values[offset+i];
    printf("%10d %10.3f %10.3f %10.3f\n",offset+i,test_chi2[offset+i],chi2->values[offset+i],diff);
  }

  double tol = 1.e-8;
  int ca = 0; // counter above threshold
  int cb = 0; // counter below threshold
  for(int i=0;i<Nloc*Nloc;i++){
    double diff = test_chi2[i] - chi2->values[i];
    if( abs(diff) > tol ){
      ca++;
    } else if( abs(diff) > 0.0 ){
      cb++;
    }
  }
  double pc_a = (double) 100.0*ca/(Nloc*Nloc); // percentage above threshold
  double pc_b = (double) 100.0*cb/(Nloc*Nloc); // percentage below threshold
  printf("Diff. above / below tolerance (%e) out of %d chi2 values: %d (%6.2f%%) / %d (%6.2f%%)\n",tol,Nloc*Nloc,ca,pc_a,cb,pc_b);
  
  free(test_chi2);
}



/*
SUMMARY:

INPUT:
\param M -- Vector of masses [in units of solar mass].

FUNCTION:

OUTPUT:
*/
void test_setup_integral(Chi2SortBins* sort_gpu,Chi2SortBins* sort_cpu,int offset,int N){
  int Nloc = sort_gpu->Nloc;
  
  // Fetch GPU sorted indices
  unsigned int* test_ind = (unsigned int*) malloc(Nloc*Nloc*sizeof(unsigned int));
  cudaMemcpy(test_ind,sort_gpu->d_sorted_ind,Nloc*Nloc*sizeof(unsigned int),cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Print GPU and CPU chi2 terms and their difference
  printf("%10s %10s %10s %10s\n","index","GPU","CPU","diff");
  for(int i=0;i<N;i++){
    unsigned int diff = test_ind[offset+i] - sort_cpu->sorted_ind[offset+i];
    printf("%10d %10d %10d %10d\n",offset+i,test_ind[offset+i],sort_cpu->sorted_ind[offset+i],diff);
  }

  double tol = 1;
  int ca = 0; // counter above threshold
  for(int i=0;i<Nloc*Nloc;i++){
    unsigned int diff = test_ind[i] - sort_cpu->sorted_ind[i];
    if( diff > tol ){
      ca++;
    }
  }
  double pc_a = (double) 100.0*ca/(Nloc*Nloc); // percentage above threshold
  printf("Diff. above tolerance (%e) out of %d indices: %d (%6.2f%%)\n",tol,Nloc*Nloc,ca,pc_a);

  free(test_ind);
}












/*
SUMMARY:
Sets the sorted indices of the Nloc*Nloc pairs according to their z, the lower_ind and n_per_bin arrays for the binned z and chi2, and bins z. 

INPUT:
\param LCA -- A filled instance of SimLC for image A.
\param LCB -- Same for image B.  
\param mpd_ratio -- An instance of Mpd that holds the probability density of the ratio of magnifications between images A and B. We need this to get the bin limits. 
\param sort_struct -- An empty instance of Chi2SortBins that will be filled by this function. 

FUNCTION:
The first step is to calculate the z that correspond to the Nloc*Nloc pairs that form between the sampled locations from images A and B.  
This step is using the same "offset-squares" algorithm as the 'calculate_chi2_GPU' function below to avoid thread racing for read-access to the same memory locations.
Then we use a double-sort to obtain the indices that will map thread IDs to the correct chi2 memory index (filling d_sorted_ind of the output Chi2SortBins object).
Because this array has a 1-to-1 correspondance with the chi2 array, these indices can be used to sort accordingly the chi2 array too.
The sorted z are used together with the bins' right edges (from mpd_ratio) to obtain the upper_bound indices of the z array and the number of values in each bin.
After a simple step to convert the upper to lower indices, this fills the lower_ind and n_per_bins arrays of the output Chi2SortBins object.
Because this array has a 1-to-1 correspondance with the chi2 array, these arrays can be used to bin accordingly the (sorted) chi2 array too.

OUTPUT:
The filled given Chi2SortBins object.

NOTES:
We could probably use a smarter approach to find the correct mapping of thread IDs to the sorted-by-z memory index, e.g. thrust library's permutation operators.
We need to check whether the fact that the memory look-up for the sorted indices leads to a performance penalty due to non-local memory access by the kernel_chi2 threads.
We can probably improve the way the lower indices are calculated for a slightly cleaner code.
*/
void setup_integral_GPU(SimLC* LCA,SimLC* LCB,Mpd* mpd_ratio,Chi2SortBins* sort_struct){
  int Nloc = LCA->Nloc;
  int Nprof = LCA->Nprof;
  
  double* d_z;
  cudaMalloc(&d_z,Nloc*Nloc*sizeof(double));


  // ############################################## Calculate z ###############################################################
  int Nblocks = (int) ceil((double) Nloc/1024.0);
  for(int k=0;k<Nblocks;k++){
    kernel_z<<<Nblocks,1024>>>(k,Nloc,Nprof,LCA->d_LC,LCB->d_LC,d_z);
    cudaDeviceSynchronize();
  }
  // ##########################################################################################################################


  // ############################################## Sort z and get sorted indices #############################################
  // This part can be probably achieved by using thrust::permutation iterators to avoid double sorting
  thrust::device_vector<int> d_tmp(Nloc*Nloc);
  thrust::sequence(d_tmp.begin(),d_tmp.end());
  thrust::device_ptr<double> d_ptr = thrust::device_pointer_cast(d_z);
  thrust::stable_sort_by_key(d_ptr,d_ptr+Nloc*Nloc,d_tmp.begin());

  thrust::device_ptr<unsigned int> ptr_sorted = thrust::device_pointer_cast(sort_struct->d_sorted_ind);
  thrust::sequence(ptr_sorted,ptr_sorted+Nloc*Nloc);
  thrust::stable_sort_by_key(d_tmp.begin(),d_tmp.end(),ptr_sorted);
  d_tmp.clear();
  d_tmp.shrink_to_fit();
  // ##########################################################################################################################

  
  // ############################################## Bin z and keep lower indices and counts ###################################
  int Nbins = mpd_ratio->Nbins;
  
  thrust::device_vector<double> d_bins(mpd_ratio->bins,mpd_ratio->bins+Nbins);
  thrust::device_vector<unsigned int> d_upper(Nbins);
  thrust::upper_bound(d_ptr,d_ptr+Nloc*Nloc,d_bins.begin(),d_bins.end(),d_upper.begin());
  unsigned int* ptr_d_upper = thrust::raw_pointer_cast(&d_upper[0]);
  cudaMemcpy(sort_struct->lower_ind+1,ptr_d_upper,(Nbins-1)*sizeof(unsigned int),cudaMemcpyDeviceToHost);
  sort_struct->lower_ind[0] = 0;
  
  thrust::device_vector<unsigned int> d_n(Nbins);
  thrust::adjacent_difference(d_upper.begin(),d_upper.end(),d_n.begin());
  unsigned int* ptr_d_n = thrust::raw_pointer_cast(&d_n[0]);
  cudaMemcpy(sort_struct->n_per_bin,ptr_d_n,Nbins*sizeof(unsigned int),cudaMemcpyDeviceToHost);
  // ##########################################################################################################################


  cudaFree(d_z);
}



/*
SUMMARY:
Calculates the chi2 values and uses the sorted indices to store them directly in the right order (sorted by z).

INPUT:
\param chi2_vars -- An instance of Chi2Vars, that containes all the pre-computed variables that facilitate the chi2 calculation.
\param chi2_d_values -- A pointer to the 'd_values' variable from an empty instance of Chi2 - this is a pointer to GPU memory.
\param sort_struct -- An filled instance of Chi2SortBins, from which we use the 'd_sorted_ind' variable.
\param LCA -- A filled instance of SimLC for image A, from which we use the d_LC and d_DLC arrays (in GPU memory).
\param LCB -- Same for image B.  

FUNCTION:
This function is using the "offset-squares" algorithm to combine the Nloc light curves from images A and B into pairs.
Its main goal is to avoid thread racing for memory read-access (although this read-access should still be ok as opposed to write access). 
See also the description of 'kernel_chi2'.

OUTPUT:
The chi2_d_values array is filled with the chi2 values. The order is dictated by the sorted indices of the corresponding z values.

NOTES:
Some variables necessary for the chi2 calculation, e.g. the data, uncertainty, interpolation indices etc, are copied into register memory.
We have to be careful not to run out of this memory that has a specific limit (~64kB).
*/
void calculate_chi2_GPU(Chi2Vars* chi2_vars,double* chi2_d_values,Chi2SortBins* sort_struct,SimLC* LCA,SimLC* LCB){
  int N = chi2_vars->Njp;
  int Nloc = LCA->Nloc;
  int Nprof = LCA->Nprof;
  
  // ############################################## Memory Allocation on the GPU ##############################################
  printf("Used GPU memory:  %d (Mb)\n",(int) get_used_gpu_mem());

  // Transfer the fixed arrays to the GPU
  cudaMemcpyToSymbol(d_d,chi2_vars->new_d,N*sizeof(double),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_s,chi2_vars->new_s,N*sizeof(double),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_facA,chi2_vars->facA,N*sizeof(double),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_facB,chi2_vars->facB,N*sizeof(double),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_indA,chi2_vars->indA,N*sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_indB,chi2_vars->indB,N*sizeof(int),0,cudaMemcpyHostToDevice);
  printf("Allocated CONSTANT memory: all arrays of size %d - data <double>, uncertainty <double>, factors <double> (x2), and indices <int> (x2): %ld (bytes)\n",N,(4*sizeof(double)+2*sizeof(int))*N);

  printf("Used GPU memory:  %d (Mb)\n",(int) get_used_gpu_mem());
  cudaDeviceSynchronize();
  // ##########################################################################################################################

  
  // Call chi2 kernel
  int Nblocks = (int) ceil((double) Nloc/1024.0);
  for(int k=0;k<Nblocks;k++){
    kernel_chi2<<<Nblocks,1024>>>(k,N,Nprof,LCA->d_LC,LCB->d_LC,LCA->d_DLC,LCB->d_DLC,Nloc,chi2_d_values,sort_struct->d_sorted_ind);
    cudaDeviceSynchronize();
  }
}




/*
SUMMARY:
Bins the computed and sorted chi2 values (size of Nloc*Nloc) into the provided bins (size of Nbins << Nloc*Nloc).

INPUT:
\param binned_chi2 -- A pointer to where the y-values of the bins will be stored for the chi2 (size of Nbins).
\param binned_exp -- A pointer to where the y-values of the bins will be stored for the exp(-0.5*chi2) (size of Nbins).
\param sort_struct -- A filled instance of Chi2SortBins. Here we need the lower_ind, n_per_bin, and Nbins variables.
\param chi2_d_values -- A pointer to the 'd_values' variable from a filled instance of Chi2 - this is a pointer to GPU memory.

FUNCTION:
Required memory for the lower_ind and n_per_bin is allocated on the GPU, as well as for the output variables binned_chi2 and binned_exp.
A grid of 'kernel_bin_chi2' kernels is launched (see its documentation).

OUTPUT:
The average of the chi2 and exp(-0.5*chi2) values that fall in each bin in the binned_chi2 and binned_exp arrays

NOTES:
The lower_ind and n_per_bin, which both have a size of Nbins, are copied to register memory that is limited (~64kb).
But, this shouldn't be a problem because realistically we expect Nbins to be <1000.
For the same reason, we expect one, at most two, blocks of 1024 threads to be created. 
*/
void bin_chi2_GPU(double* binned_chi2,double* binned_exp,Chi2SortBins* sort_struct,double* chi2_d_values){
  int Nloc = sort_struct->Nloc;
  int Nbins = sort_struct->Nbins;
  
  // Transfer the fixed arrays to the GPU
  cudaMemcpyToSymbol(d_lower_ind,sort_struct->lower_ind,Nbins*sizeof(unsigned int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_n_per_bin,sort_struct->n_per_bin,Nbins*sizeof(unsigned int),0,cudaMemcpyHostToDevice);
  double* d_binned_chi2;
  cudaMalloc((void**) &d_binned_chi2,Nbins*sizeof(double));
  double* d_binned_exp;
  cudaMalloc((void**) &d_binned_exp,Nbins*sizeof(double));
  
  int Nblocks = (int) ceil((double) Nloc/1024.0);
  kernel_bin_chi2<<<Nblocks,1024>>>(Nbins,d_binned_chi2,d_binned_exp,chi2_d_values);
  cudaDeviceSynchronize();

  cudaMemcpy(binned_chi2,d_binned_chi2,Nbins*sizeof(double),cudaMemcpyDeviceToHost);  
  cudaMemcpy(binned_exp,d_binned_exp,Nbins*sizeof(double),cudaMemcpyDeviceToHost);  
  cudaFree(d_binned_chi2);
  cudaFree(d_binned_exp);
}



/*
SUMMARY:
Performs a Fourier transform.

INPUT:
\param plan -- The predefined CUFFT plan.
\param data_GPU -- The array on the GPU that will be transformed.
\param Fdata_GPU -- The complex array that will hold the Fourier transform result. 

FUNCTION:
Performs a Fourier transform.

OUTPUT:
The Fourier transform in the Fdata_GPU array.
*/
void myfft2d_r2c(cufftHandle* plan,cufftDoubleReal* data_GPU,cufftDoubleComplex* Fdata_GPU){
  // Do the fourier transform on the GPU
  cufftResult result = cufftExecD2Z(*plan, data_GPU, Fdata_GPU);
  if (cuda_error("Cuda error: Failed to execut plan", false, __FILE__, __LINE__)) std::runtime_error("CUFFT Error: unable to execute plan");
}



/*
SUMMARY:
Performs an inverse Fourier transform.

INPUT:
\param plan -- The predefined CUFFT plan.
\param Fdata_GPU -- The complex array on the GPU that will be transformed.
\param data_GPU -- The real array that will hold the inverse Fourier transform result. 

FUNCTION:
Performs an inverse Fourier transform.

OUTPUT:
The inverse Fourier transform in the data_GPU array.
*/
void myfft2d_c2r(cufftHandle* plan, cufftDoubleComplex* Fdata_GPU, cufftDoubleReal* data_GPU){
  // Do the inverse fourier transform on the GPU
  cufftResult result = cufftExecZ2D(*plan, Fdata_GPU, data_GPU);
  if (cuda_error("Cuda error: Failed to execut plan", false, __FILE__, __LINE__)) std::runtime_error("CUFFT Error: unable to execute plan");
}



/*
SUMMARY:
Multiplying the Fourier transformed map and kernel.

INPUT:
\param Fmap -- The Fourier transform of the map (array of complex numbers).
\param Fkernel -- The Fourier transform of the kernel (array of complex numbers).
\param norm -- The normalization factor equal to the total number of pixels in a map.

FUNCTION:
Multiplies the Fourier transforms of the map and the kernel in the correct way.

OUTPUT:
The real and imaginary parts of each pixel are stored 'in-place' in the array of the Fourier transform of the kernel.

NOTES:
If we manage to normalize the result of this kernel properly, then we can switch from double to single precision. 
*/
__global__ void kernel_multiplyFFTs(cufftDoubleComplex* Fmap,cufftDoubleComplex* Fkernel,double norm){
  unsigned long int i = (blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x + threadIdx.x; // thread ID
  cufftDoubleReal dum1 = (cufftDoubleReal) (Fmap[i].x*Fkernel[i].x - Fmap[i].y*Fkernel[i].y);
  cufftDoubleReal dum2 = (cufftDoubleReal) (Fmap[i].x*Fkernel[i].y + Fmap[i].y*Fkernel[i].x);
  Fkernel[i].x = dum1/norm;
  Fkernel[i].y = dum2/norm;
}



/*
SUMMARY:
Samples the convolved map and stores the magnification values at the correct light curve position.

INPUT:
\param convmap -- The convolved map.
\param d_LC -- An array that holds the magnification values for each light curve (size of Nloc*Nprof).
\param d_DLC -- An array that holds the adjacent difference between light curve values (size of Nloc*(Nprof-1)).
\param k -- the index of the profile for which the convolved map is being sampled. 
\param Nprof -- The number of profiles, i.e. the length of the light curves.
\param loc_x -- The x-pixel indices of the Nloc sampled locations.
\param loc_y -- The y-pixel indices of the Nloc sampled locations.
\param Nx -- The width of the map in pixels, need to combine correctly the x and y pixel indices.
\param Nloc -- The number of locations that are sampled.

FUNCTION:
Each thread access a pair of x and y pixel indices and samples the convmap at that location.
The value is stored at the right position, as defined by the profile index k, in the light curve array d_LC.
If that position is >=1 then the adjacent difference with the previous value of the light curve is calculated too.

OUTPUT:
The arrays d_LC and d_DLC containing the magnification values for each light curve and their adjacent difference are filled. 
*/
__global__ void kernel_sample(cufftDoubleReal* convmap,double* d_LC,double* d_DLC,int k,int Nprof,int* loc_x,int* loc_y,int Nx,int Nloc){
  // Stores the light curve in sizes of Nprof
  //unsigned int id = threadIdx.x;
  unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;
  double val;
  if( id<Nloc ){
    unsigned int i = loc_y[id];
    unsigned int j = loc_x[id];
    unsigned int index = i*Nx+j;
    val = convmap[index];
    d_LC[id*Nprof+k] = val;

    if( k>0 ){
      d_DLC[id*(Nprof-1)+k-1] = val - d_LC[id*Nprof+k-1];
    }
  }
}



/*
SUMMARY:
Calculates the z values on the GPU.

INPUT:
\param loop_ind -- The index of the loop of the "offset-squares" algorithm.
\param Nloc -- The number of sampled locations per image, i.e. the number of light curves.
\param Nprof -- The size of each light curve, i.e. the number of source profiles.
\param d_LCA -- Light curve values for image A (size of Nloc*Nprof).
\param d_LCB -- Light curve values for image B (size of Nloc*Nprof).
\param d_z -- The array of z, i.e. the ratios of the point-source magnifications of each light curve (the first entry in each light curve) with size Nloc*Nloc.

FUNCTION:
This kernel adopts the same grid structure with the extra loop as described for the "offset-squares" algorithm (see 'kernel_chi2').
Each thread is structured similarly as well, i.e. it reads the first value from a light curve from image A and then loops over all light curves of image B ensuring no overlap in reading.

OUTPUT:
The point-source magnification ratios between images A and B that correspond to the sampled light curves in array d_z (size of Nloc*Nloc). 
*/
__global__ void kernel_z(int loop_ind,int Nloc,int Nprof,double* d_LCA,double* d_LCB,double* d_z){
  int Nt  = blockDim.x; // equal to 1024
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int a = bid*Nt + tid;
  int b;
  
  for(int j=0;j<Nt;j++){
    b = ((bid+loop_ind)%gridDim.x)*Nt + (tid+j)%Nt;
    
    if( a < Nloc && b < Nloc ){
      d_z[a*Nloc+b] = d_LCA[a*Nprof]/d_LCB[b*Nprof];
    }
  }
  __syncthreads();
}



/*
SUMMARY:
Calculates the chi2 values on the GPU.

INPUT:
\param loop_ind -- The index of the loop of the "offset-squares" algorithm.
\param N -- The size of the data.
\param Nprof -- The size of each light curve, i.e. the number of source profiles.
\param d_LCA -- Light curve values for image A (size of Nloc*Nprof).
\param d_LCB -- Light curve values for image B (size of Nloc*Nprof).
\param d_DLCA -- Adjacent difference between light curve values for image A (size of Nloc*(Nprof-1)).
\param d_DLCB -- Adjacent difference between light curve values for image B (size of Nloc*(Nprof-1)).
\param Nloc -- The number of sampled locations per image, i.e. the number of light curves.
\param chi2_d_values -- A pointer to the 'd_values' variable from an empty instance of Chi2 - this is a pointer to GPU memory.
\param sorted -- The indices that map thread IDs to the correct memory location of chi2, i.e. sorted according to z.

FUNCTION:
This kernel implements the "offset-squares" algorithm to avoid its threads racing for read-access from the same memory locations (although this should be ok anyway).
Each block consists of 1024 threads (at most) that create the pairs between 1024 light curves from image A and 1024 from image B.
Each thread begins by reading a single light curve from image A specified by its thread ID.
Then it loops over all the 1024 light curves of image B, calculates their ratio, and computes the chi2 square value.
But, to ensure that each thread will read a different B light curve, the loop starting index is offset by its thread ID and wrapped when it is beyond the 1024 image B light curves. 
At the end of each iteration the threads are synchronized and move together to reading the next light curve B, which ensures that they will all again read different B light curves.
Each computed chi2 value is stored at the 'sorted by z' memory location through the indices 'sorted'.
Regarding the grid of blocks, it must be similarly structured to avoid reading from the same memory locations for either light curves A or B.
We setup a 2D grid of blocks, each covering 1024 A and B light curves that shouldn't overlap between the blocks.
To achieve this, we should parse this 2D grid in a specific way: Imagine starting by the main diagonal of this 2D grid, which will ensure no overlap between the blocks, and then moving to the next diagonal by wrapping around the edges of the 2D grid.
In fact, this results in a 1D grid and a loop that calls the kernel with different block specifications each time.
This is a short description of the "offset-squares" algorithm.

OUTPUT:
The computed chi2 values are stored in the provided chi2_d_values pointer, a pointer to GPU memory from a Chi2 instance.
Because we use the indices 'sorted' from an instance of Chi2SortBins, which map the thread IDs to the ordered memory location of the chi2 array (sorted according to the corresponding z), the chi2 values are stored in the right order.

NOTES:
This kernel requires the following variables to be in register memory: interpolation indices and factors for images A and B, data and uncertainties.
Also, a light curve from image A is read and kept during the execution of the kernel, which means that there should be enough register memory to do that.
*/
__global__ void kernel_chi2(int loop_ind,int N,int Nprof,double* d_LCA,double* d_LCB,double* d_DLCA,double* d_DLCB,int Nloc,double* chi2_d_values,unsigned int* sorted){
  int Nt  = blockDim.x; // equal to 1024
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int a = bid*Nt + tid;
  double chi2,mB,tmp;
  int b,sorted_index;
  double mA[50];

  if( a < Nloc ){     
    // Get all the values of mA just once for each thread
    for(int i=0;i<N;i++){
      // Interpolate magnification A
      if( d_indA[i] == -1 ){
	mA[i] = 1;
      } else {
	//mA[i] = LCA[a*Nprof+d_indA[i]] + d_facA[i]*( LCA[a*Nprof+d_indA[i]+1] - LCA[a*Nprof+d_indA[i]] );
	mA[i] = d_LCA[a*Nprof+d_indA[i]] + d_facA[i]*d_DLCA[a*(Nprof-1)+d_indA[i]];
      }
    }
  }
  __syncthreads();
    
    
  for(int j=0;j<Nt;j++){
    b = ((bid+loop_ind)%gridDim.x)*Nt + (tid+j)%Nt;

    if( a < Nloc && b < Nloc ){
      chi2 = 0.0;
      for(int i=0;i<N;i++){
	// Interpolate magnification B
	if( d_indB[i] == -1 ){
	  mB = 1;
	} else {
	  //mB = LCB[b*Nprof+d_indB[i]] + d_facB[i]*( LCB[b*Nprof+d_indB[i]+1] - LCB[b*Nprof+d_indB[i]] );
	  mB = d_LCB[b*Nprof+d_indB[i]] + d_facB[i]*d_DLCB[b*(Nprof-1)+d_indB[i]];
	}
	
	// Calculate chi2 term
	tmp = (d_d[i] - (mA[i]/mB))/d_s[i];
	chi2 += tmp*tmp;
      }
      sorted_index = sorted[a*Nloc+b];
      chi2_d_values[sorted_index] = chi2;
      //chi2_d_values[a*Nloc+b] = chi2;
    }
    
    __syncthreads();
  }

}



/*
SUMMARY:
Biining the Nloc*Nloc sorted chi2 and corresponding exp(0.5*chi2) values.

INPUT:
\param Nbins -- the number of bins that we will bin the chi2 and exp(0.5*chi2) values into.
\param d_binned_chi2 -- GPU memory to hold the chi2 bins.
\param d_binned_exp -- GPU memory to hold the exp(0.5*chi2) bins.
\param d_chi2 -- GPU memory for the sorted Nloc*Nloc chi2 values.

FUNCTION:
Each thread performs a loop over the N chi2 values that fall in each bin.
Some bins are empty so the threads won't do anything.
The loop calculates the average of chi2 and exp(-0.5*chi2) values.

OUTPUT:
The average chi2 and exp(-0.5*chi2) values per bin are stored in the d_binned_chi2 and d_binned_exp arrays.
*/
__global__ void kernel_bin_chi2(int Nbins,double* d_binned_chi2,double* d_binned_exp,double* d_chi2){
  int Nt  = blockDim.x; // equal to 1024
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int id = bid*Nt+tid;
  double sum_exp,sum_chi;

  if( id<Nbins ){
    if( d_n_per_bin[id] > 0 ){
      sum_exp = 0.0;
      sum_chi = 0.0;
      for(int i=0;i<d_n_per_bin[id];i++){
	sum_exp += exp(-0.5*d_chi2[d_lower_ind[id]+i]);
	sum_chi += d_chi2[d_lower_ind[id]+i];
      }
      d_binned_exp[id]  = sum_exp/d_n_per_bin[id];
      d_binned_chi2[id] = sum_chi/d_n_per_bin[id];
    } else {
      d_binned_exp[id] = exp(-2000.0/2.0);
      d_binned_chi2[id] = 2000.0;
    }
  }
  
}

