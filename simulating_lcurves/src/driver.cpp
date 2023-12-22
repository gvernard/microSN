#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "json/json.h"

#include "magnification_map.hpp"
#include "locations.hpp"
#include "data_structs.hpp"
#include "gpu_functions.hpp"
#include "cpu_functions.hpp"
#include "util.hpp"


int main(int argc,char* argv[]){

  // INITIALIZATION
  //******************************************************************************************
  //Json::Value::Members jmembers;

  // Read the main projection parameters
  Json::Value input;
  std::ifstream fin;
  fin.open("input.json",std::ifstream::in);
  fin >> input;
  fin.close();


  // Define parameters and perform checks for LC simulator code 
  int Nprof = input["sizes"].size();
  std::vector<double> sizes(Nprof);
  for(int i=0;i<Nprof;i++){
    sizes[i] = input["sizes"][i].asDouble();
  }  
  std::string shape = input["shape"].asString();
  printf("Number of profiles set to: %d (%s)\n",Nprof,shape.c_str());
  
  std::string locations = input["locations"]["type"].asString();
  int Nloc;
  int* x;
  int* y;
  if( locations == "grid" ){
    int Nside = input["locations"]["Nside"].asInt();
    Nloc = Nside*Nside;
    x = (int*) malloc(Nloc*sizeof(int));
    y = (int*) malloc(Nloc*sizeof(int));
    int Nmap = input["locations"]["Nmap"].asInt();
    double ss = input["locations"]["ss"].asDouble();
    int Noff = (int) ceil((Nmap/ss)*sizes.back());
    locations_on_grid(Nside,Nmap,Noff,x,y);
  } else {
    Nloc = input["locations"]["Nloc"].asInt();
    x = (int*) malloc(Nloc*sizeof(int));
    y = (int*) malloc(Nloc*sizeof(int));
    locations_random(Nloc,x,y);
  }
  printf("Sampling from %d locations per map (%s).\n",Nloc,locations.c_str());
  printf("Will loop %.2f times (round up) to cover all location pairs.\n",Nloc/1024.0);


  // Data structures that will not change for different sampled parameters
  SimLC LCA(Nloc,Nprof);
  SimLC LCB(Nloc,Nprof);
  Chi2 chi2(Nloc);


  
  // Define parameters and perform checks for chi2 and likelihood code 
  int Nd = input["data"][0]["d"].size();
  try {
    if( Nd >= 50 ){
      throw std::runtime_error("Number of data points is too large to fit in register memory!");
    } else {
      printf("Data length: %d\n",Nd);
    }
  } catch(std::exception &e) {
    std::cerr << e.what() << std::endl;
    return false;
  }
  double d[Nd];
  double s[Nd];
  double t[Nd];
  for(int i=0;i<Nd;i++){
    d[i] = input["data"][0]["d"][i].asDouble();
    s[i] = input["data"][0]["s"][i].asDouble();
    t[i] = input["data"][0]["t"][i].asDouble();
  }
  double DL = input["DL"].asDouble();
  double DS = input["DS"].asDouble();
  double DLS = input["DLS"].asDouble();
  double Dfac = sqrt(DS*DLS/DL);

  long int total = get_total_gpu_mem();
  printf("Total GPU memory: %ld (bytes)\n",total);


  // Define parameters for the MPDs
  int Nbins_mpd  = input["mpds"]["mpd"]["Nbins"].asInt();
  double min_mpd = input["mpds"]["mpd"]["min"].asDouble();
  double max_mpd = input["mpds"]["mpd"]["max"].asDouble();
  int Nbins_ratio  = input["mpds"]["ratio"]["Nbins"].asInt();
  double min_ratio = input["mpds"]["ratio"]["min"].asDouble();
  double max_ratio = input["mpds"]["ratio"]["max"].asDouble();
  //******************************************************************************************



  
  // BEGIN: SAMPLE PER MAP PAIR
  //============================================================================================================

  // We read the magnification map stored in the gerlumph format (map.bin and map_meta.dat)
  double dum_rein = 1.0;
  MagnificationMap mapA("12123",dum_rein);
  MagnificationMap mapB("11156",dum_rein);


  
  //******************************************************************************************
  std::cout << "CONVOLUTIONS" << std::endl;
  std::cout << std::string(100, '*') << std::endl;

  // Call the LC simulator code for first map
  expanding_source(&mapA,sizes,shape,Nloc,x,y,&LCA);
  
  // Call the LC simulator code for second map
  expanding_source(&mapB,sizes,shape,Nloc,x,y,&LCB);
  //******************************************************************************************


  
  //******************************************************************************************
  std::cout << "MPDS" << std::endl;
  std::cout << std::string(100, '*') << std::endl;

  // Calculate the probability distribution of magnification ratios z  
  Mpd binned_mpd_A = mapA.getBinnedMpd(Nbins_mpd,min_mpd,max_mpd);
  Mpd binned_mpd_B = mapB.getBinnedMpd(Nbins_mpd,min_mpd,max_mpd);
  binned_mpd_A.normalize();
  binned_mpd_B.normalize();

  Mpd ratio(Nbins_ratio,min_ratio,max_ratio,"log");
  binned_mpd_A.divide_by(&binned_mpd_B,&ratio);
  //std::cout << ratio.bins[0] << "   " << ratio.bins[Nbins_ratio-1] << std::endl;
  
  binned_mpd_A.writeMpd("mpd_A.dat");
  binned_mpd_B.writeMpd("mpd_B.dat");
  ratio.writeMpd("mpd_R.dat");

  unsigned int* sorted    = (unsigned int*) malloc(Nloc*Nloc*sizeof(unsigned int));
  unsigned int* upper_ind = (unsigned int*) malloc(Nbins_ratio*sizeof(unsigned int));
  unsigned int* n_per_bin = (unsigned int*) malloc(Nbins_ratio*sizeof(unsigned int));
  setup_integral(&LCA,&LCB,&ratio,sorted,upper_ind,n_per_bin);
  
  std::cout << "Integral A: " << binned_mpd_A.integrate() << std::endl;
  std::cout << "Integral B: " << binned_mpd_B.integrate() << std::endl;
  std::cout << "Integral ratio: " << ratio.integrate() << std::endl;
  //******************************************************************************************




  std::cout << "CHI-SQUARED" << std::endl;
  std::cout << std::string(100, '*') << std::endl;
  
  // BEGIN: SAMPLE MASS AND VELOCITY
  //============================================================================================================
  std::vector<double> M{0.4,0.5}; // in solar masses                  SAMPLED
  std::vector<double> V{1}; // in 10^5 km/s                           SAMPLED

  Chi2Vars chi2_vars = setup_chi2_calculation(M,V,Dfac,Nd,d,t,s,Nprof,sizes.data());
  calculate_chi2_GPU(&chi2_vars,&chi2,&LCA,&LCB);


  // Calcuate the chi2 on the CPU too and compare values
  LCA.transfer_to_CPU(); // Needed to calculate the chi2 on the CPU
  LCB.transfer_to_CPU(); // Needed to calculate the chi2 on the CPU
  calculate_chi2_CPU(&chi2_vars,&chi2,&LCA,&LCB);

  int test_offset = 23*Nloc+24;
  int test_size = 10;
  test_chi2(&chi2,test_offset,test_size);


  //******************************************************************************************
 
  // Sort and bin chi2 values with respect to z
  
  // Calculate integral using the binned chi2 and p(z)

  
  // OUTPUT
  /*
  FILE* fh = fopen("lcurves.dat","w");
  for(int i=0;i<Nloc;i++){
    for(int j=0;j<Nprof;j++){
      fprintf(fh," %f",LCA[i*Nprof+j]);
    }
    fprintf(fh,"\n");
  }
  fclose(fh);
  */
  //******************************************************************************************


  //============================================================================================================
  // END: SAMPLE MASS AND VELOCITY

  



  free(sorted);
  free(upper_ind);
  free(n_per_bin);
  //============================================================================================================
  // END: SAMPLE MAP PAIR


  
  return 0;
}
