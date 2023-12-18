#include <cmath>
#include <cufft.h>
#include <iostream>

#include "variability_models.hpp"
#include "magnification_map.hpp"
#include "profile.hpp"
#include "util.hpp"


static int myfft2d_r2c(cufftHandle* plan, cufftDoubleReal* data, cufftDoubleComplex* Fdata);
static int myfft2d_c2r(cufftHandle* plan, cufftDoubleComplex* Fdata, cufftDoubleReal* data);
__global__ void kernelMultiplyFFTs(cufftDoubleComplex* Fmap,cufftDoubleComplex* Fkernel,double norm);
__global__ void sampleConvmap(cufftDoubleReal* convmap,double* LC,int k,int Nprof,int* loc_x,int* loc_y,int Nx,int Nloc);


/*
INPUT:
 1. File location for the magnification map
 2. A vector<double> of the profile half-light radii in units of RE (length of Nprof)
 3. A vector<int> of pixel indices to sample from (length of Nloc)
 4. The shape of the profile (could easily be a vector of time-dependent profiles)


FUNCTION:
First, some memory needs to be allocated on the GPU:
1. to read in a magnification map.
2. to store the convolution kernel (changes per loop iteration, see below).
   There may be extra memory needed to store intermediate products of the convolution.
3. An 'LC' array of size Nprof x Nloc to store the light curves.

Inside a loop over the profile sizes it:
- creates the profile according to the given shape (CPU)
- creates the corredponding kernel at the predefined memory location (GPU)
- performs the convolution (GPU)
- samples the convolved map at the given pixels and stores them in the LC array (GPU)


OUTPUT:
The LC array.


NOTES: 
1. The final LC arrays should persist in GPU memory after this code quits - how can we achieve this?
2. This code should ensure early on that there is enough memory on the GPU for it to run.
3. We could transfer the sampled pixel values to the CPU after each convolution.
   This will allow for the highest possible number of Nloc to be sampled.
*/

void expanding_source(std::string map_id,std::vector<double> sizes,std::string shape,int Nloc,int* sample_loc_x,int* sample_loc_y,double* LC){
  int Nprof = sizes.size();
  dim3 block_mult(1000); // Nx/f
  dim3 grid_mult(10,5001); // f,Ny/2+1
  dim3 block_samp(1024);
  dim3 grid_samp((int) ceil(Nloc/1024));
  
  
  // We read the magnification map stored in the gerlumph format (map.bin and map_meta.dat)
  double dum_rein = 1.0;
  gerlumph::MagnificationMap map(map_id,dum_rein);
  int Nx = map.Nx;
  int Ny = map.Ny;
  double norm = Nx*Ny;
  //map.writeImageFITS("map.fits",10);
  //gerlumph::MagnificationMap dum_map = map; // A test map on the CPU

  
  // Calculate the maximum offset in pixels
  double max_size = sizes.back(); // in units of Rein
  int maxOffset = (int) ceil( (Nx/map.width)*max_size );
  gerlumph::Kernel kernel(map.Nx,map.Ny);



  // Create profile parameters
  std::map<std::string,std::string> profile_pars;
  profile_pars.insert(std::pair<std::string,std::string>("shape", shape));
  profile_pars.insert(std::pair<std::string,std::string>("rhalf", "dum"));
  profile_pars.insert(std::pair<std::string,std::string>("pixSizePhys", std::to_string(map.pixSizePhys)));
  profile_pars.insert(std::pair<std::string,std::string>("incl", "0"));
  profile_pars.insert(std::pair<std::string,std::string>("orient", "0"));

  
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

  // Allocate memory: for the final LC array
  double* d_LC;
  cudaMalloc(&d_LC,Nprof*Nloc*sizeof(double));
  printf("Allocated memory: Light curves, Nprof*Nloc <double>: %d (Mb)\n",Nprof*Nloc*8);

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
  // ##########################################################################################################################


  

  // ############################################## Operations of the GPU #####################################################
  // Do the Fourier transform of the emap and store it on the GPU
  cudaMemcpy( any_map_GPU, map.data, Nx*Ny*sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);
  myfft2d_r2c(&plan_r2c,any_map_GPU,Fmap_GPU);

  // Loop over the kernels
  for(int k=0;k<Nprof;k++){
    std::cout << "Profile: " << k << std::endl;

    // Create profile and kernel
    profile_pars["rhalf"] = std::to_string(sizes[k]*map.pixSizePhys*map.Nx/map.width);
    gerlumph::BaseProfile* profile = gerlumph::FactoryProfile::getInstance()->createProfileFromHalfRadius(profile_pars);
    kernel.setKernel(profile);
    delete(profile);

    // Fourier transform of kernel
    cudaMemcpy( any_map_GPU, kernel.data, Nx*Ny*sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);
    myfft2d_r2c(&plan_r2c,any_map_GPU,Fkernel_GPU);

    // Multiply Fourier transforms of map and kernel
    kernelMultiplyFFTs<<<grid_mult,block_mult>>>(Fmap_GPU,Fkernel_GPU,norm);
    cudaDeviceSynchronize();

    // Get inverse Fourier transform of product
    myfft2d_c2r(&plan_c2r,Fkernel_GPU,any_map_GPU);

    // Transfer convolved map to CPU and write image.
    //cudaMemcpy(dum_map.data,convmap_GPU,Nx*Ny*sizeof(double),cudaMemcpyDeviceToHost);
    //dum_map.writeImageFITS("conv_"+std::to_string(k)+".fits",10);

    // Sample convolved map
    sampleConvmap<<<grid_samp,block_samp>>>(any_map_GPU,d_LC,k,Nprof,loc_x,loc_y,Nx,Nloc);
  }
  // ##########################################################################################################################



  
  cudaMemcpy(LC,d_LC,Nprof*Nloc*sizeof(double),cudaMemcpyDeviceToHost);



  
  // ############################################## Cleanup ###################################################################
  cudaFree(Fkernel_GPU);
  cudaFree(Fmap_GPU);
  cudaFree(any_map_GPU);
  cudaFree(loc_x);
  cudaFree(loc_y);
  cudaFree(d_LC);
  cufftDestroy(plan_r2c);
  cufftDestroy(plan_c2r);
  // ##########################################################################################################################
}



int myfft2d_r2c(cufftHandle* plan,cufftDoubleReal* data_GPU,cufftDoubleComplex* Fdata_GPU){
  // Do the fourier transform on the GPU
  cufftResult result = cufftExecD2Z(*plan, data_GPU, Fdata_GPU);
  if (cuda_error("Cuda error: Failed to execut plan", false, __FILE__, __LINE__)) std::runtime_error("CUFFT Error: unable to execute plan");
  cudaDeviceSynchronize();
  return 0;
}


int myfft2d_c2r(cufftHandle* plan, cufftDoubleComplex* Fdata_GPU, cufftDoubleReal* data_GPU){
  // Do the inverse fourier transform on the GPU
  cufftResult result = cufftExecZ2D(*plan, Fdata_GPU, data_GPU);
  if (cuda_error("Cuda error: Failed to execut plan", false, __FILE__, __LINE__)) std::runtime_error("CUFFT Error: unable to execute plan");
  cudaDeviceSynchronize();
  return 0;
}


__global__ void kernelMultiplyFFTs(cufftDoubleComplex* Fmap,cufftDoubleComplex* Fkernel,double norm){
  unsigned long int i = (blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x + threadIdx.x; // thread ID
  cufftDoubleReal dum1 = (cufftDoubleReal) (Fmap[i].x*Fkernel[i].x - Fmap[i].y*Fkernel[i].y);
  cufftDoubleReal dum2 = (cufftDoubleReal) (Fmap[i].x*Fkernel[i].y + Fmap[i].y*Fkernel[i].x);
  Fkernel[i].x = dum1/norm;
  Fkernel[i].y = dum2/norm;
}


__global__ void sampleConvmap(cufftDoubleReal* convmap,double* LC,int k,int Nprof,int* loc_x,int* loc_y,int Nx,int Nloc){
  // Stores the light curve in sizes of Nprof
  //unsigned int id = threadIdx.x;
  unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;
  if( id<Nloc ){
    unsigned int i = loc_y[id];
    unsigned int j = loc_x[id];
    unsigned int index = i*Nx+j;
    LC[id*Nprof+k] = convmap[index];
  }
}








