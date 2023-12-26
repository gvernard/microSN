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

__global__ void kernel_chi2(int loop_ind,int N,int Nprof,double* d_LCA,double* d_LCB,double* d_DLCA,double* d_DLCB,int Nloc,double* chi2_all,unsigned int* sorted);
__global__ void kernel_z(int k,int Nloc,int Nprof,double* d_LCA,double* d_LCB,double* d_z);
__global__ void kernel_multiplyFFTs(cufftDoubleComplex* Fmap,cufftDoubleComplex* Fkernel,double norm);
__global__ void kernel_sample(cufftDoubleReal* convmap,double* d_LC,double* d_DLC,int k,int Nprof,int* loc_x,int* loc_y,int Nx,int Nloc);
__global__ void kernel_bin_chi2(int Nloc,int Nbins,double* d_binned_chi2,double* d_binned_exp,double* chi2);
void myfft2d_r2c(cufftHandle* plan, cufftDoubleReal* data, cufftDoubleComplex* Fdata);
void myfft2d_c2r(cufftHandle* plan, cufftDoubleComplex* Fdata, cufftDoubleReal* data);


/*
INPUT:
 1. File location/ID for the magnification map
 2. A vector<double> of the profile half-light radii in units of RE (length of Nprof)
 3. The shape of the profile (could easily be a vector of time-dependent profiles)
 4. The number of sampled locations (Nloc)
 5. Two arrays of the x and y pixel indices to sample from (length of Nloc)
 6. A pointer to the CPU memory to store the output - size of Nprof*Nloc
 6. A pointer to the CPU memory to store the adjacent difference output - size of (Nprof-1)*Nloc


FUNCTION:
First, some memory needs to be allocated on the GPU:
1. To store the Fourier transform of the magnification map.
2. To store the Fourier transform of the convolution kernel (changes per loop iteration, see below).
3. To store the convolved map. This memory is also used to read-in the map and kernel before transforming them.
4. There is extra memory needed to run the CUFFT plans.
5. An 'LC' array of size Nprof x Nloc to store the light curves.

First, we get the Fourier transform of the map and store it on the GPU.
Inside a loop over the profile sizes we:
- create the profile according to the given shape
- create the corredponding kernel at the predefined memory location
- get the Fourier transform of the kernel
- multiply it with the Fourier transform of the map
- get inverse Fourier transform of the product
- sample the convolved map at the given pixels and store them in the LC array


OUTPUT:
The LC array with size Nloc*Nprof.
The array of differences between adjacent profiles for each light curve with size Nloc*(Nprof-1).

NOTES: 
- This code should ensure early on that there is enough memory on the GPU for it to run.
- The first profile is ignored, that's why it's R1/2 should always be 0. The unconvolved maps are sampled instead.
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


void calculate_chi2_GPU(Chi2Vars* chi2_vars,Chi2* chi2,Chi2SortBins* sort_struct,SimLC* LCA,SimLC* LCB){
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
    kernel_chi2<<<Nblocks,1024>>>(k,N,Nprof,LCA->d_LC,LCB->d_LC,LCA->d_DLC,LCB->d_DLC,Nloc,chi2->d_values,sort_struct->d_sorted_ind);
    cudaDeviceSynchronize();
  }
}


void bin_chi2_GPU(int Nloc,double* binned_chi2,double* binned_exp,Chi2SortBins* sort_struct,Chi2* chi2){
  int Nbins = sort_struct->Nbins;
  
  // Transfer the fixed arrays to the GPU
  cudaMemcpyToSymbol(d_lower_ind,sort_struct->lower_ind,Nbins*sizeof(unsigned int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_n_per_bin,sort_struct->n_per_bin,Nbins*sizeof(unsigned int),0,cudaMemcpyHostToDevice);
  double* d_binned_chi2;
  cudaMalloc((void**) &d_binned_chi2,Nbins*sizeof(double));
  double* d_binned_exp;
  cudaMalloc((void**) &d_binned_exp,Nbins*sizeof(double));
  
  int Nblocks = (int) ceil((double) Nloc/1024.0);
  kernel_bin_chi2<<<Nblocks,1024>>>(Nloc,Nbins,d_binned_chi2,d_binned_exp,chi2->d_values);
  cudaDeviceSynchronize();

  cudaMemcpy(binned_chi2,d_binned_chi2,Nbins*sizeof(double),cudaMemcpyDeviceToHost);  
  cudaMemcpy(binned_exp,d_binned_exp,Nbins*sizeof(double),cudaMemcpyDeviceToHost);  
  cudaFree(d_binned_chi2);
  cudaFree(d_binned_exp);
}


void test_chi2(Chi2* chi2,int offset,int N,int Nloc){
  
  // Fetch some GPU chi2 terms and print them
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



void setup_integral(SimLC* LCA,SimLC* LCB,Mpd* mpd_ratio,Chi2SortBins* sort_struct){
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


  // ############################################## Sort z and get indices ####################################################
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

  
  // ############################################## Bin z and keep indices and counts #########################################
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

  // for(int i=0;i<Nbins;i++){
  //   printf("Bin %d:  %f >   %d  N=%d\n",i,mpd_ratio->bins[i],sort_struct->upper_ind[i],sort_struct->n_per_bin[i]);
  // }
  // ##########################################################################################################################


  cudaFree(d_z);
}

void sort_chi2_by_z_GPU(int Nloc,Chi2SortBins* sort_struct,Chi2* chi2){
  // thrust::device_ptr<unsigned int> ptr_sorted = thrust::device_pointer_cast(sort_struct->d_sorted_ind);
  // thrust::device_ptr<double> d_ptr = thrust::device_pointer_cast(chi2->d_values);
  // thrust::sort_by_key(ptr_sorted,ptr_sorted + Nloc*Nloc,d_ptr);

  double* tmp = (double*) malloc(Nloc*Nloc*sizeof(double));
  cudaMemcpy(tmp,chi2->d_values,Nloc*Nloc*sizeof(double),cudaMemcpyDeviceToHost);
  unsigned int* ind = (unsigned int*) malloc(Nloc*Nloc*sizeof(unsigned int));
  cudaMemcpy(ind,sort_struct->d_sorted_ind,Nloc*Nloc*sizeof(unsigned int),cudaMemcpyDeviceToHost);
  
  for(int i=0;i<Nloc*Nloc;i++){
    chi2->values[i] = tmp[ind[i]];
  }
  cudaMemcpy(chi2->d_values,chi2->values,Nloc*Nloc*sizeof(double),cudaMemcpyHostToDevice);

  free(tmp);
  free(ind);
}

void myfft2d_r2c(cufftHandle* plan,cufftDoubleReal* data_GPU,cufftDoubleComplex* Fdata_GPU){
  // Do the fourier transform on the GPU
  cufftResult result = cufftExecD2Z(*plan, data_GPU, Fdata_GPU);
  if (cuda_error("Cuda error: Failed to execut plan", false, __FILE__, __LINE__)) std::runtime_error("CUFFT Error: unable to execute plan");
}


void myfft2d_c2r(cufftHandle* plan, cufftDoubleComplex* Fdata_GPU, cufftDoubleReal* data_GPU){
  // Do the inverse fourier transform on the GPU
  cufftResult result = cufftExecZ2D(*plan, Fdata_GPU, data_GPU);
  if (cuda_error("Cuda error: Failed to execut plan", false, __FILE__, __LINE__)) std::runtime_error("CUFFT Error: unable to execute plan");
}


__global__ void kernel_multiplyFFTs(cufftDoubleComplex* Fmap,cufftDoubleComplex* Fkernel,double norm){
  unsigned long int i = (blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x + threadIdx.x; // thread ID
  cufftDoubleReal dum1 = (cufftDoubleReal) (Fmap[i].x*Fkernel[i].x - Fmap[i].y*Fkernel[i].y);
  cufftDoubleReal dum2 = (cufftDoubleReal) (Fmap[i].x*Fkernel[i].y + Fmap[i].y*Fkernel[i].x);
  Fkernel[i].x = dum1/norm;
  Fkernel[i].y = dum2/norm;
}


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

__global__ void kernel_chi2(int loop_ind,int N,int Nprof,double* d_LCA,double* d_LCB,double* d_DLCA,double* d_DLCB,int Nloc,double* chi2_all,unsigned int* sorted){
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
      chi2_all[sorted_index] = chi2;
      //chi2_all[a*Nloc+b] = chi2;
    }
    
    __syncthreads();
  }

}


__global__ void kernel_bin_chi2(int Nloc,int Nbins,double* d_binned_chi2,double* d_binned_exp,double* chi2){
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
	sum_exp += exp(-0.5*chi2[d_lower_ind[id]+i]);
	sum_chi += chi2[d_lower_ind[id]+i];
      }
      d_binned_exp[id]  = sum_exp/d_n_per_bin[id];
      d_binned_chi2[id] = sum_chi/d_n_per_bin[id];
    } else {
      d_binned_exp[id] = exp(-2000.0/2.0);
      d_binned_chi2[id] = 2000.0;
    }
  }
  
}

