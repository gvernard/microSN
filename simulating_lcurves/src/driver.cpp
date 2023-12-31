#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "json/json.h"

#include "magnification_map.hpp"
#include "gpu_functions.hpp"
#include "cpu_functions.hpp"
#include "data_structs.hpp"
#include "integration.hpp"
#include "locations.hpp"
#include "util.hpp"


int main(int argc,char* argv[]){

  // INITIALIZATION
  //******************************************************************************************
  //Json::Value::Members jmembers;

  if( argc != 2 ){
    std::cout << "Error: You need to provide the name of a .json file that contains all the input variables!" << std::endl;
    return 1;
  }
  
  // Read the main projection parameters
  Json::Value input;
  std::ifstream fin;
  fin.open(argv[1],std::ifstream::in);
  fin >> input;
  fin.close();

  //bool compare_cpu = true;
  bool compare_cpu = false;

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
    int Nmap = input["locations"]["Nmap"].asInt();
    double ss = input["locations"]["ss"].asDouble();
    int Noff = (int) ceil((Nmap/ss)*sizes.back());
    int seed = input["locations"]["seed"].asInt();
    locations_random(Nloc,Nmap,Noff,seed,x,y);
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

  FILE* fh;
  FILE* fh2;

  long int total = get_total_gpu_mem();
  printf("Total GPU memory: %ld (bytes)\n",total);


  // Define parameters for the MPDs
  int Nbins_mpd  = input["mpds"]["mpd"]["Nbins"].asInt();
  double min_mpd = input["mpds"]["mpd"]["min"].asDouble();
  double max_mpd = input["mpds"]["mpd"]["max"].asDouble();
  int Nbins_ratio  = input["mpds"]["ratio"]["Nbins"].asInt();
  double min_ratio = input["mpds"]["ratio"]["min"].asDouble();
  double max_ratio = input["mpds"]["ratio"]["max"].asDouble();

  Chi2SortBins sort_struct(Nloc,Nbins_ratio);
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

  // Need to transfer the light curves to the CPU for comparisons
  if( compare_cpu ){
    LCA.transfer_to_CPU();
    LCB.transfer_to_CPU();
  }
  //******************************************************************************************


  
  //******************************************************************************************
  std::cout << "MPDS" << std::endl;
  std::cout << std::string(100, '*') << std::endl;

  Mpd binned_mpd_A = mapA.getBinnedMpd(Nbins_mpd,min_mpd,max_mpd);
  Mpd binned_mpd_B = mapB.getBinnedMpd(Nbins_mpd,min_mpd,max_mpd);
  binned_mpd_A.normalize();
  binned_mpd_B.normalize();

  Mpd ratio(Nbins_ratio,min_ratio,max_ratio,"log");
  binned_mpd_A.divide_by(&binned_mpd_B,&ratio);
  
  binned_mpd_A.writeMpd("mpd_A.dat");
  binned_mpd_B.writeMpd("mpd_B.dat");
  ratio.writeMpd("mpd_R.dat");

  setup_integral_GPU(&LCA,&LCB,&ratio,&sort_struct);

  if( compare_cpu ){
    Chi2SortBins sort_struct_cpu(Nloc,Nbins_ratio);
    setup_integral_CPU(&LCA,&LCB,&ratio,&sort_struct_cpu);

    int test_offset = Nloc*Nloc-10;
    int test_size = 10;
    test_setup_integral(&sort_struct,&sort_struct_cpu,test_offset,test_size);
  }
  // std::cout << "Integral A: " << binned_mpd_A.integrate() << std::endl;
  // std::cout << "Integral B: " << binned_mpd_B.integrate() << std::endl;
  // std::cout << "Integral ratio: " << ratio.integrate() << std::endl;
  //******************************************************************************************



  std::cout << "CHI-SQUARED" << std::endl;
  std::cout << std::string(100, '*') << std::endl;
  
  // BEGIN: SAMPLE MASS AND VELOCITY
  //============================================================================================================
  std::vector<double> M{0.4,0.5}; // in solar masses                  SAMPLED
  std::vector<double> V{1}; // in 10^5 km/s                           SAMPLED
  double* binned_chi2_GPU = (double*) malloc(Nbins_ratio*sizeof(double));
  double* binned_exp_GPU  = (double*) malloc(Nbins_ratio*sizeof(double));

  
  Chi2Vars chi2_vars = setup_chi2_calculation(M,V,Dfac,Nd,d,t,s,Nprof,sizes.data());
  calculate_chi2_GPU(&chi2_vars,chi2.d_values,&sort_struct,&LCA,&LCB);

  if( compare_cpu ){
    sort_struct.transfer_to_CPU();
    std::cout << "Sorted indices transfered to CPU" << std::endl;
    calculate_chi2_CPU(&chi2_vars,chi2.values,&sort_struct,&LCA,&LCB);
    
    //int test_offset = 123*Nloc+24;
    int test_offset = Nloc*Nloc-10;
    int test_size = 10;
    test_chi2(&chi2,test_offset,test_size);
  }

 
  bin_chi2_GPU(binned_chi2_GPU,binned_exp_GPU,&sort_struct,chi2.d_values);
  double integral_gpu = trapezium(ratio.Nbins,binned_exp_GPU,ratio.counts,ratio.bins);
  //double integral_gpu = simpson(ratio.Nbins,binned_exp_GPU,ratio.counts,ratio.bins);

  if( compare_cpu ){
    chi2.transfer_to_CPU();
    double* binned_chi2_CPU = (double*) malloc(Nbins_ratio*sizeof(double));
    double* binned_exp_CPU  = (double*) malloc(Nbins_ratio*sizeof(double));
    bin_chi2_CPU(binned_chi2_CPU,binned_exp_CPU,&sort_struct,chi2.values);    
    double integral_cpu = trapezium(ratio.Nbins,binned_exp_CPU,ratio.counts,ratio.bins);
    //double integral_cpu = simpson(ratio.Nbins,binned_exp_CPU,ratio.counts,ratio.bins);
    std::cout << "GPU: " << integral_gpu << "  CPU: " << integral_cpu << std::endl;
    free(binned_chi2_CPU);
    free(binned_exp_CPU);
  } else {
    std::cout << "GPU: " << integral_gpu << std::endl;
  }

  double like = integral_gpu*exp(-0.5*chi2_vars.const_term);
  

  // OUTPUT: Write binned exp(0.5*chi2) and number of values per bin 
  fh  = fopen("binned_exp.dat","w");
  fh2 = fopen("binned_N.dat","w");
  for(int i=0;i<sort_struct.Nbins;i++){
    fprintf(fh,"%11.6e %11.6e\n",ratio.bins[i],binned_exp_GPU[i]);
    fprintf(fh2,"%11.6e %d\n",ratio.bins[i],sort_struct.n_per_bin[i]);
  }
  fclose(fh);
  fclose(fh2);
  
  //============================================================================================================
  // END: SAMPLE MASS AND VELOCITY

  



  // OUTPUT: Write light curves A and B
  fh  = fopen("lcurvesA.dat","w");
  fh2 = fopen("lcurvesB.dat","w");
  for(int i=0;i<Nloc;i++){
    for(int j=0;j<Nprof;j++){
      fprintf(fh," %f",LCA.LC[i*Nprof+j]);
      fprintf(fh2," %f",LCB.LC[i*Nprof+j]);
    }
    fprintf(fh,"\n");
    fprintf(fh2,"\n");
  }
  fclose(fh);
  fclose(fh2);  
  //============================================================================================================
  // END: SAMPLE MAP PAIR


  free(binned_chi2_GPU);
  free(binned_exp_GPU);
  
  return 0;
}
