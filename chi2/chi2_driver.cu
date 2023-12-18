#include <cstdlib>
#include <iostream>

#include "chi2_functions.cuh"
#include "util.cuh"


int main(int argc, char* argv[])
{
  
  /*******************************************************************************
  N -- number of lightcurves
  F -- number of filters (wavelengths)
  T -- number of timestamps
  *******************************************************************************/
  int N = 10000;
  int F = 7;
  int T = 10;

  float* f_obs = nullptr;
  cudaMallocManaged((void**)&f_obs, F * T * sizeof(float));
  if (cuda_error("cudaMallocManaged(*f_obs)", false, __FILE__, __LINE__)) return -1;
  float* df_obs = nullptr;
  cudaMallocManaged((void**)&df_obs, F * T * sizeof(float));
  if (cuda_error("cudaMallocManaged(*df_obs)", false, __FILE__, __LINE__)) return -1;

  float* mu1 = nullptr;
  cudaMallocManaged((void**)&mu1, N * F * T * sizeof(float));
  if (cuda_error("cudaMallocManaged(*mu1)", false, __FILE__, __LINE__)) return -1;
  float* mu2 = nullptr;
  cudaMallocManaged((void**)&mu2, N * F * T * sizeof(float));
  if (cuda_error("cudaMallocManaged(*mu2)", false, __FILE__, __LINE__)) return -1;
  

  // Defining fake data (actual values don't matter)
  for (int f = 0; f < F; f++)
  {
    for (int t = 0; t < T; t++)
    {
      f_obs[f * T + t] = 0.97;
      df_obs[f * T + t] = 1.1;
    }
  }

  // Defining fake N simulated light curves (actual values don't matter)
  for (int n = 0; n < N; n++)
  {
    for (int f = 0; f < F; f++)
    {
      for (int t = 0; t < T; t++)
      {
        mu1[n * F * T + f * T + t] = 0.42;
        mu2[n * F * T + f * T + t] = 1.1;
      }
    }
  }

  float* likelihood = nullptr;
  cudaMallocManaged((void**)&likelihood, sizeof(float));
  if (cuda_error("cudaMallocManaged(*likelihood)", false, __FILE__, __LINE__)) return -1;

  *likelihood = 0;

  std::cout << "GPU run:" << std::endl;
  getChi2GPU(f_obs, df_obs, mu1, mu2, N, F, T, likelihood);
  std::cout << "Likelihood: " << *likelihood << std::endl;

  std::size_t free = get_free_gpu_mem();
  std::cout << "number of free bytes in gpu memory: " << free << "\n";

  *likelihood = 0;

  std::cout << "CPU run:" << std::endl;
  getChi2CPU(f_obs, df_obs, mu1, mu2, N, F, T, likelihood);
  std::cout << "Likelihood: " << *likelihood << std::endl;

  return 0;

}
