#include "chi2_functions.hpp"


__global__ void kernelChi2(float* d_mua, float* d_mub, float* d_like_dum, int Nwave, int Nlocs);
__constant__ float const_f_obs[64];
__constant__ float const_df_obs[64];



/*******************************************************************************
calculate the chi-squared likelihood on the GPU

\param like_all -- pointer to array of likelihoods
\param f_obs_all -- pointer to array of observation flux ratios
                    assumes array is of the form [filter1, ... filterNwave]
                    where each filter is of the form [loc1, ... locNlocs]
\param df_obs -- pointer to array of errors on the observations
                 assumes a single value for the error per filter
                 i.e. it is of the form [filter1, ... filterNwave]
\param h_mua -- pointer to array of simulated fluxes for image 1
                see f_obs_all for its structure
\param h_mub -- pointer to array of simulated fluxes for image 2
                see f_obs_all for its structure
\param Nlocs -- number of locations simulated for each image.
                assumed to be the same for image 1 and image 2
\param Nwave -- number of wavelengths (filters)
\param Nratios -- number of observations (timestamps)
*******************************************************************************/
void getChi2Cuda(double* like_all, float** f_obs_all, float* df_obs, float** h_mua, float** h_mub, int Nlocs, int Nwave, int Nratios)
{
  cudaError_t err;

  // Size of the shared memory required in the kernel is always Nwave * Nthreads * sizeof(float)
  // i.e. this is the chunk of muA locations for all wavelengths.
  // I need to generally keep this below 48KB.
  int Ngrid;
  int Nthreads;
  //ensure that grid and block size are such that memory is below 30KB
  setGridThreads(Ngrid, Nthreads, Nwave); 
  printf("Shared memory occupancy: %d / 48000\n", Nwave * Nthreads * sizeof(float));
  printf("Ngrid / Nthreads = %d / %d\n", Ngrid, Nthreads);


  // Allocate memory on device
  // inputs:
  float* d_mua;
  float* d_mub;
  cudaMalloc(&d_mua, Nwave * Nlocs * sizeof(float));
  cudaMalloc(&d_mub, Nwave * Nlocs * sizeof(float));

  // output:
  int Nlike = Ngrid * Nthreads;
  float* h_like_dum = (float*) malloc(Nlike * sizeof(float));
  float* d_like_dum;
  cudaMalloc(&d_like_dum, Nlike * sizeof(float));
  err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    fprintf(stderr, "Error: %s - in \"memory allocation\" \n", cudaGetErrorString(err));
  }

  // Transfer (input) memory from host to device
  for(int k = 0; k < Nwave; k++)
  {
    cudaMemcpy(d_mua + k * Nlocs, h_mua[k], Nlocs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mub + k * Nlocs, h_mub[k], Nlocs * sizeof(float), cudaMemcpyHostToDevice);
  }
  //assumes we have 64 or less filters (see definition of const_df_obs at start of file)
  cudaMemcpyToSymbol(const_df_obs, (void*)df_obs, Nwave * sizeof(float), 0);
  err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    fprintf(stderr, "Error: %s - in \"memory transfer to device\" \n", cudaGetErrorString(err));
  }


  // Loop over f_obs (Nratios times)
  dim3 grid(Ngrid);
  dim3 threads(Nthreads);
  int Nmem = Nwave * Nthreads;
  //for every timestamp
  for(int i = 0; i < Nratios; i++)
  {
    // Transfer f_obs to constant memory
    //assumes we have 64 or less filters (see definition of const_f_obs at start of file)
    cudaMemcpyToSymbol(const_f_obs, (void*)f_obs_all[i], Nwave * sizeof(float), 0);

    // Execute GPU kernels
    kernelChi2<<<grid, threads, Nmem * sizeof(float)>>>(d_mua, d_mub, d_like_dum, Nwave, Nlocs);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
      fprintf(stderr,"Error: %s - in \"kernelChi2\" \n", cudaGetErrorString(err));
    }

    // Get memory from device
    cudaMemcpy(h_like_dum, d_like_dum, Nlike * sizeof(float), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
      fprintf(stderr,"Error: %s - in \"memory transfer from device\" \n", cudaGetErrorString(err));
    }

    // Add all the likelihoods in h_like_dum
    double like = 0.0;
    for(int j = 0; j < Nlike; j++)
    {
      like += h_like_dum[j];
    }
    like_all[i] = like;

    std::cout << like_all[i] << std::endl;
  }

  free(h_like_dum);
}



/*******************************************************************************
calculate the chi-squared likelihood on the GPU for a single timestamp

\param d_mua -- pointer to array of simulated fluxes for image 1
\param d_mub -- pointer to array of simulated fluxes for image 2
\param d_like_dum -- pointer to array of likelihoods
\param Nwave -- number of wavelengths (filters)
\param Nlocs -- number of locations simulated for each image.
                assumed to be the same for image 1 and image 2
*******************************************************************************/
__global__ void kernelChi2(float* d_mua, float* d_mub, float* d_like_dum, int Nwave, int Nlocs)
{
  unsigned int t         = threadIdx.x;
  unsigned int Nthreads  = blockDim.x;
  unsigned int thread_id = blockIdx.x * Nthreads + t;
  unsigned int Nblocks   = gridDim.x;

  // Allocate shared memory
  extern __shared__ float mua[];

  //this uses float, but GPUs are capable of double precision now
  //not sure if the greater precision from doubles really matters to us though
  float like = 0.0;
  // Loop over all of muA, reading it block by block
  for(int j = 0; j < Nblocks; j++)
  {
    // Each thread reads Nwave entries from muA into shared memory (different wavelengths from the same magmap location)
    for(int k = 0; k < Nwave; k++)
    {
      mua[k * Nthreads + t] = d_mua[k * Nlocs + Nthreads * j + t];
    }
    __syncthreads();

    // Each thread combines its unique muB value (thread_id) with the muA values currently into shared memory
    for(int i = 0; i < Nthreads; i++)
    {
      float fac = 0.0;
      for(int k = 0; k < Nwave; k++)
      {
        float fsim = mua[k * Nthreads + i]/d_mub[k * Nlocs + thread_id];
        float dum = (const_f_obs[k] - fsim) / const_df_obs[k];
        fac += dum * dum;
      }
      like += exp(-fac / 2);
    }
  }
  d_like_dum[thread_id] = like;
}



/*******************************************************************************
set the number of blocks in the grid, and the number of threads in a block,
for a given number of filters
ensures that grid and block size are such that memory in a block is below 30KB

\param Ngrid -- reference to number of blocks per grid
\param Nthreads -- reference to number of threads per block
\param Nwave -- number of filters (wavelengths)
*******************************************************************************/
void setGridThreads(int& Ngrid, int& Nthreads, int Nwave)
{
  Ngrid = 10;
  Nthreads = 1000;
  int shared_mem_size = Nthreads * Nwave * sizeof(float);

  while (shared_mem_size > 30000)
  {
    Ngrid *= 2;
    Nthreads /= 2;
    shared_mem_size = Nthreads * Nwave * sizeof(float);
  }
}



/*******************************************************************************
calculate the chi-squared likelihood on the CPU

\param like_all -- pointer to array of likelihoods
\param f_obs_all -- pointer to array of observation flux ratios
\param df_obs -- pointer to array of errors on the observations
\param h_mua -- pointer to array of simulated fluxes for image 1
\param h_mub -- pointer to array of simulated fluxes for image 2
\param Nlocs -- number of locations simulated for each image.
                assumed to be the same for image 1 and image 2
\param Nwave -- number of wavelengths (filters)
\param Nratios -- number of observations (timestamps)
*******************************************************************************/
void getChi2CudaCPU(double* like_all, float** f_obs_all, float* df_obs, float** h_mua, float** h_mub, int Nlocs, int Nwave, int Nratios)
{

  //for every timestamp
  for(int q = 0; q < Nratios; q++)
  {
    double like = 0.0;
    //for every simulated lightcurve
    for(int i = 0; i < Nlocs; i++)
    {
      for(int j = 0; j < Nlocs; j++)
      {
        double chi2 = 0.0;
        //for every filter
        for(int k = 0; k < Nwave; k++)
        {
          double fsim = h_mua[k][i] / h_mub[k][j];
          double dum = (f_obs_all[q][k] - fsim) / df_obs[k];
          chi2 += dum * dum;
        }
        //add to the likelihood for this timestamp
        //should we calculate the log likelihood instead to avoid
        //floating point precision loss?
        //in addition, it's not immediately obvious that adding the likelihood from
        //every filter together is the right thing to do per timestamp
        //although after some thought, so long as the simulations account for the different
        //sizes per time stamp, maybe it is?
        like += exp(-chi2 / 2);
      }
    }
    //set the likelihood for this timestamp
    like_all[q] = like;
    std::cout << like_all[q] << std::endl;
  }
}

