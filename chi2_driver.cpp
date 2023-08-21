#include <cstdlib>
#include <iostream>

#include "functions.hpp"


int main(int argc, char* argv[]){
  int Nratios = 10;
  int Nlocs = 10000;
  int Nwave = 7;




  double* like_all = (double*) malloc(Nratios*sizeof(double));
  float** f_obs_all = (float**) malloc(Nratios*sizeof(float*));
  for(int i=0;i<Nratios;i++){
    f_obs_all[i] = (float*) malloc(Nwave*sizeof(float));
  }
  float* df_obs = (float*) malloc(Nwave*sizeof(float));


  float** mua = (float**) malloc(Nwave*sizeof(float*));
  float** mub = (float**) malloc(Nwave*sizeof(float*));
  for(int i=0;i<Nwave;i++){
    mua[i] = (float*) malloc(Nlocs*sizeof(float));
    mub[i] = (float*) malloc(Nlocs*sizeof(float));
  }  



  for(int j=0;j<Nratios;j++){
    for(int i=0;i<Nwave;i++){
      f_obs_all[j][i] = 0.97;
    }
  }
  for(int i=0;i<Nwave;i++){
    df_obs[i] = 1.1;
  }

  for(int i=0;i<Nwave;i++){
    for(int j=0;j<Nlocs;j++){
      mua[i][j] = 0.42;
      mub[i][j] = 1.1;
    }
  }


  std::cout << "GPU run:" << std::endl;
  getChi2Cuda(like_all,f_obs_all,df_obs,mua,mub,Nlocs,Nwave,Nratios);


  std::cout << "CPU run:" << std::endl;
  getChi2CudaCPU(like_all,f_obs_all,df_obs,mua,mub,Nlocs,Nwave,Nratios);





  free(like_all);
  for(int i=0;i<Nratios;i++){
    free(f_obs_all[i]);
  }
  free(f_obs_all);
  free(df_obs);

  for(int i=0;i<Nwave;i++){
    free(mua[i]);
    free(mub[i]);
  }
  free(mua);
  free(mub);

  return 0;
}
