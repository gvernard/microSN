#include <cmath>
#include <cufft.h>
#include <iostream>

#include "variability_models.hpp"
#include "magnification_map.hpp"
#include "profile.hpp"


static int myfft2d_r2c(int Nx, int Ny, cufftDoubleReal* data, cufftDoubleComplex* Fdata);
static int myfft2d_c2r(int Nx, int Ny, cufftDoubleComplex* Fdata, cufftDoubleReal* data);
__global__ void kernelMultiplyFFTs(cufftDoubleComplex* Fmap,cufftDoubleComplex* Fkernel,double norm);
__global__ void sampleConvmap(cufftDoubleReal* convmap,double* LC,int k,int Nprof,int* loc_x,int* loc_y,int Nx);


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
  dim3 block(1000); // Nx/f
  dim3 block_grid(10,5001); // f,Ny/2+1
  
  
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
  // Allocate memory: for the sampled locations
  int* loc_x;
  cudaMalloc(&loc_x,Nloc*sizeof(int));
  cudaMemcpy(loc_x,sample_loc_x,Nloc*sizeof(int),cudaMemcpyHostToDevice);
  int* loc_y;
  cudaMalloc(&loc_y,Nloc*sizeof(int));
  cudaMemcpy(loc_y,sample_loc_y,Nloc*sizeof(int),cudaMemcpyHostToDevice);

  // Allocate memory: for the Fourier transform of the map on the GPU (COMPLEX)
  cufftDoubleComplex* Fmap_GPU;
  cudaMalloc( (void**) &Fmap_GPU, Nx*(Ny/2+1)*sizeof(cufftDoubleComplex));
  if( cudaGetLastError() != cudaSuccess ){
    fprintf(stderr, "Cuda error: Failed to allocate Fmap_GPU\n");
    throw std::bad_alloc();
  }

  // Allocate memory: for the Fourier transform of the kernel on the GPU (COMPLEX)
  cufftDoubleComplex* Fkernel_GPU;
  cudaMalloc( (void**) &Fkernel_GPU, Nx*(Ny/2+1)*sizeof(cufftDoubleComplex));
  if( cudaGetLastError() != cudaSuccess ){
    fprintf(stderr, "Cuda error: Failed to allocate Fkernel_GPU\n");
    throw std::bad_alloc();
  }

  // Allocate memory: for the convolved map on the GPU (REAL)
  cufftDoubleReal* convmap_GPU;
  cudaMalloc((void**) &convmap_GPU, Nx*Ny*sizeof(cufftDoubleReal));
  if( cudaGetLastError() != cudaSuccess ){
    fprintf(stderr, "Cuda error: Failed to allocate convmap_GPU\n");
    throw std::bad_alloc();
  }

  // Allocate memory: for the final LC array
  double* d_LC;
  cudaMalloc(&d_LC,Nprof*Nloc*sizeof(double));
  // ##########################################################################################################################


  

  // ############################################## Operations of the GPU #####################################################
  // Do the Fourier transform of the emap and store it on the GPU
  myfft2d_r2c(map.Nx,map.Ny,map.data,Fmap_GPU);

  // Loop over the kernels
  for(int k=0;k<Nprof;k++){
    std::cout << "Profile: " << k << std::endl;

    // Create profile and kernel
    profile_pars["rhalf"] = std::to_string(sizes[k]*map.pixSizePhys*map.Nx/map.width);
    gerlumph::BaseProfile* profile = gerlumph::FactoryProfile::getInstance()->createProfileFromHalfRadius(profile_pars);
    kernel.setKernel(profile);
    delete(profile);

    // Fourier transform of kernel
    myfft2d_r2c(kernel.Nx,kernel.Ny,kernel.data,Fkernel_GPU);

    // Multiply Fourier transforms of map and kernel
    kernelMultiplyFFTs<<<block_grid,block>>>(Fmap_GPU,Fkernel_GPU,norm);
    cudaDeviceSynchronize();

    // Get inverse Fourier transform of product
    myfft2d_c2r(map.Nx,map.Ny,Fkernel_GPU,convmap_GPU);

    // Transfer convolved map to CPU and write image.
    //cudaMemcpy(dum_map.data,convmap_GPU,Nx*Ny*sizeof(double),cudaMemcpyDeviceToHost);
    //dum_map.writeImageFITS("conv_"+std::to_string(k)+".fits",10);

    // Sample convolved map
    sampleConvmap<<<1,Nloc>>>(convmap_GPU,d_LC,k,Nprof,loc_x,loc_y,Nx);
  }
  // ##########################################################################################################################



  
  cudaMemcpy(LC,d_LC,Nprof*Nloc*sizeof(double),cudaMemcpyDeviceToHost);
  cudaFree(Fkernel_GPU);
  cudaFree(Fmap_GPU);
  cudaFree(convmap_GPU);
  cudaFree(loc_x);
  cudaFree(loc_y);
  cudaFree(d_LC);
}



int myfft2d_r2c(int Nx,int Ny,cufftDoubleReal* data,cufftDoubleComplex* Fdata_GPU){
  cufftResult result;
  cufftHandle plan;
  cufftDoubleReal* data_GPU;

  //allocate and transfer memory to the GPU
  cudaMalloc( (void**) &data_GPU, Nx*Ny*sizeof(cufftDoubleReal));
  if( cudaGetLastError() != cudaSuccess ){
    fprintf(stderr, "Cuda error: Failed to allocate data_GPU\n");
    throw std::bad_alloc();
  }
  cudaMemcpy( data_GPU, data, Nx*Ny*sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);

  //do the fourier transform on the GPU
  result = cufftPlan2d(&plan,Nx,Ny,CUFFT_D2Z);
  if( result != CUFFT_SUCCESS ){
    fprintf(stderr, "CUFFT Error: Unable to create plan (error code: %d)\n",result);
    cudaFree(data_GPU);
    cudaFree(Fdata_GPU);
    throw std::runtime_error("CUFFT Error: Unable to create plan");
  }
  result = cufftExecD2Z(plan, data_GPU, Fdata_GPU);
  if( result != CUFFT_SUCCESS ){
    fprintf(stderr, "CUFFT Error: unable to execute plan (error code: %d)\n",result);
    cudaFree(data_GPU);
    cudaFree(Fdata_GPU);
    cufftDestroy(plan);
    throw std::runtime_error("CUFFT Error: unable to execute plan");
  }
  cudaDeviceSynchronize();
  cufftDestroy(plan);
  cudaFree(data_GPU);

  return 0;
}



__global__ void kernelMultiplyFFTs(cufftDoubleComplex* Fmap,cufftDoubleComplex* Fkernel,double norm){
  unsigned long int i = (blockIdx.x*gridDim.y+blockIdx.y)*blockDim.x + threadIdx.x; // thread ID
  cufftDoubleReal dum1 = (cufftDoubleReal) (Fmap[i].x*Fkernel[i].x - Fmap[i].y*Fkernel[i].y);
  cufftDoubleReal dum2 = (cufftDoubleReal) (Fmap[i].x*Fkernel[i].y + Fmap[i].y*Fkernel[i].x);
  Fkernel[i].x = dum1/norm;
  Fkernel[i].y = dum2/norm;
}




int myfft2d_c2r(int Nx, int Ny, cufftDoubleComplex* Fdata_GPU, cufftDoubleReal* data_GPU){
  cufftResult result;
  cufftHandle plan;
  
  //do the inverse fourier transform on the GPU
  result = cufftPlan2d(&plan,Nx,Ny,CUFFT_Z2D) ;
  if( result != CUFFT_SUCCESS ){
    fprintf(stderr, "CUFFT Error: Unable to create plan (error code: %d)\n",result);
    cudaFree(Fdata_GPU);
    cudaFree(data_GPU);
    throw std::runtime_error("CUFFT Error: Unable to create plan");
  }
  result = cufftExecZ2D(plan, Fdata_GPU, data_GPU);
  if( result != CUFFT_SUCCESS ){
    fprintf(stderr, "CUFFT Error: unable to execute plan (error code: %d)\n",result);
    cudaFree(Fdata_GPU);
    cudaFree(data_GPU);
    cufftDestroy(plan);
    throw std::runtime_error("CUFFT Error: unable to execute plan");
  }
  cudaDeviceSynchronize();
  cufftDestroy(plan);
  
  return 0;
}


__global__ void sampleConvmap(cufftDoubleReal* convmap,double* LC,int k,int Nprof,int* loc_x,int* loc_y,int Nx){
  // Stores the light curve in sizes of Nprof
  unsigned int id = threadIdx.x;
  unsigned int i = loc_y[id];
  unsigned int j = loc_x[id];
  unsigned int index = i*Nx+j;
  LC[id*Nprof+k] = convmap[index];
}








