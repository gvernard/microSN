#pragma once

#include "stopwatch.hpp"
#include "util.cuh"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>


/*******************************************************************************
kernel to calculate the chi-squared likelihood on the GPU

\param f_obs -- pointer to array of observation flux ratios
                assumes array is of the form [filter1, ... filterF]
                where each filter is of the form [t1, ... tN]
\param df_obs -- pointer to array of errors on the observations
                 assumes array is of the form [filter1, ... filterF]
                 where each filter is of the form [t1, ... tT]
\param mu1 -- pointer to array of simulated fluxes for image 1
              assumes array is of the form [loc1, ... locN]
              where each location is of the form [filter1, ... filterF]
              where each filter is of the form [t1, ... tT]
\param mu2 -- pointer to array of simulated fluxes for image 2
              assumes array is of the form [loc1, ... locN]
              where each location is of the form [filter1, ... filterF]
              where each filter is of the form [t1, ... tT]
\param N -- number of locations simulated for each image.
            assumed to be the same for image 1 and image 2
\param F -- number of filters
\param T -- number of timestamps (observations)
\param chi2_all -- pointer to array of chi2 of size N^2
*******************************************************************************/
__global__ void kernelChi2(float* f_obs, float* df_obs, float* mu1, float* mu2, int N, int F, int T, float* chi2_all)
{

  //every x thread is a location on the magnification map for image 1
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < N; x += blockDim.x * gridDim.x)
  {

    //every y thread is a location on the magnification map for image 2
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < N; y += blockDim.y * gridDim.y)
    {

      //every z thread is a filter
      for (int f = blockIdx.z * blockDim.z + threadIdx.z; f < F; f += blockDim.z * gridDim.z)
      {
        float* filter_obs = &f_obs[f * T];
        float* filter_df_obs = &df_obs[f * T];
        float* filter_mu1 = &mu1[x * F * T + f * T];
        float* filter_mu2 = &mu2[x * F * T + f * T];

        float chi2 = 0.0f;
        //every thread loops over time sequentially
        for (int t = 0; t < T; t++)
        {
          //f_obs includes all filters, so need to include f * T to pick out the current one
          float tmp = (filter_obs[t] - filter_mu1[t] / filter_mu2[t]) / filter_df_obs[t];
          chi2 += tmp * tmp / 2;
        }
        //add chi2 value for this particular filter to the appropriate array position
        //atomic addition since every filter should be added for each pair of locations on the maps
        atomicAdd(&chi2_all[N * y + x], chi2);
      }
    }
  }
}

/*******************************************************************************
calculate the chi-squared likelihood on the GPU

\param f_obs -- pointer to array of observation flux ratios
                assumes array is of the form [filter1, ... filterF]
                where each filter is of the form [t1, ... tN]
\param df_obs -- pointer to array of errors on the observations
                 assumes array is of the form [filter1, ... filterF]
                 where each filter is of the form [t1, ... tT]
\param mu1 -- pointer to array of simulated fluxes for image 1
              assumes array is of the form [loc1, ... locN]
              where each location is of the form [filter1, ... filterF]
              where each filter is of the form [t1, ... tT]
\param mu2 -- pointer to array of simulated fluxes for image 2
              assumes array is of the form [loc1, ... locN]
              where each location is of the form [filter1, ... filterF]
              where each filter is of the form [t1, ... tT]
\param N -- number of locations simulated for each image.
            assumed to be the same for image 1 and image 2
\param F -- number of filters
\param T -- number of timestamps (observations)
\param likelihood -- pointer to likelihood
*******************************************************************************/
bool getChi2GPU(float* f_obs, float* df_obs, float* mu1, float* mu2, int N, int F, int T, float* likelihood)
{
  Stopwatch stopwatch;
  double t_elapsed;

  // Allocate memory on device
  // output:
  float* chi2_all;
  cudaMallocManaged(&chi2_all, N * N * sizeof(float));
  if (cuda_error("cudaMallocManaged(*chi2_all)", false, __FILE__, __LINE__)) return false;

  for (int i = 0; i < N * N; i++)
  {
    chi2_all[i] = 0;
  }
  
  dim3 threads;
  dim3 blocks;

  set_threads(threads, 16, 16, 2);
  set_blocks(threads, blocks, N, N, F);

  // Execute GPU kernels
  stopwatch.start();
  kernelChi2<<<blocks, threads>>>(f_obs, df_obs, mu1, mu2, N, F, T, chi2_all);
  if (cuda_error("kernelChi2", true, __FILE__, __LINE__)) return false;
  t_elapsed = stopwatch.stop(false);
  std::cout << "Elapsed time: " << t_elapsed << std::endl;

  // This summation can be replaced by a thrust reduce function
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      (*likelihood) += std::exp(-chi2_all[N * j + i]);
    }
    
  }
  t_elapsed = stopwatch.stop();
  std::cout << "Elapsed time: " << t_elapsed << std::endl;
  
  return true;
}

/*******************************************************************************
calculate the chi-squared likelihood on the CPU

\param f_obs -- pointer to array of observation flux ratios
                assumes array is of the form [filter1, ... filterF]
                where each filter is of the form [t1, ... tN]
\param df_obs -- pointer to array of errors on the observations
                 assumes array is of the form [filter1, ... filterF]
                 where each filter is of the form [t1, ... tT]
\param mu1 -- pointer to array of simulated fluxes for image 1
                assumes array is of the form [loc1, ... locN]
                where each location is of the form [filter1, ... filterF]
                where each filter is of the form [t1, ... tT]
\param mu2 -- pointer to array of simulated fluxes for image 2
                assumes array is of the form [loc1, ... locN]
                where each location is of the form [filter1, ... filterF]
                where each filter is of the form [t1, ... tT]
\param N -- number of locations simulated for each image.
            assumed to be the same for image 1 and image 2
\param F -- number of filters
\param T -- number of timestamps (observations)
\param likelihood -- pointer to likelihood
*******************************************************************************/
bool getChi2CPU(float* f_obs, float* df_obs, float* mu1, float* mu2, int N, int F, int T, float* likelihood)
{
  Stopwatch stopwatch;
  double t_elapsed;

  stopwatch.start();
  //for every location on the magnification map for image 1
  for (int x = 0; x < N; x++)
  {
    //for every location on the magnification map for image 2
    for (int y = 0; y < N; y++)
    {
      float chi2 = 0;
      //for every filter
      for (int f = 0; f < F; f++)
      {
        //for every timestamp
        for (int t = 0; t < T; t++)
        {
          float tmp = (f_obs[f * T + t] - mu1[x * F * T + f * T + t] / mu2[y * F * T + f * T + t]) / df_obs[f * T + t];
          chi2 += tmp * tmp / 2;
        }
      }
      (*likelihood) += std::exp(-chi2);
    }
  }
  t_elapsed = stopwatch.stop();
  std::cout << "Elapsed time: " << t_elapsed << std::endl;

  return true;
}

