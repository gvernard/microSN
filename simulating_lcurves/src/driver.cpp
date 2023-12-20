#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "json/json.h"
#include "variability_models.hpp"
#include "chi2_functions.hpp"
#include "locations.hpp"
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
  
  //******************************************************************************************



  
  // PROCESSING
  //******************************************************************************************

  // Call the LC simulator code for first map
  std::string map_id;
  map_id = "12115"; //                                                SAMPLED
  double* LCA = (double*) malloc(Nloc*Nprof*sizeof(double));
  double* DLCA = (double*) malloc(Nloc*(Nprof-1)*sizeof(double));
  expanding_source(map_id,sizes,shape,Nloc,x,y,LCA,DLCA);

  // Call the LC simulator code for second map
  map_id = "12115"; //                                                SAMPLED
  double* LCB = (double*) malloc(Nloc*Nprof*sizeof(double));
  double* DLCB = (double*) malloc(Nloc*(Nprof-1)*sizeof(double));
  expanding_source(map_id,sizes,shape,Nloc,x,y,LCB,DLCB);


  // Calculate differences between profiles of simulated light curves

  
  // Call the likelihood integration code
  std::vector<double> M{0.4,0.5}; // in solar masses                   SAMPLED
  std::vector<double> V{1}; // in 10^5 km/s                           SAMPLED
  double like = get_like_single_pair(M,V,Dfac,Nd,d,t,s,Nprof,sizes.data(),Nloc,LCA,LCB,DLCA,DLCB);
  //******************************************************************************************



  
  // OUTPUT
  //******************************************************************************************

  FILE* fh = fopen("lcurves.dat","w");
  for(int i=0;i<Nloc;i++){
    for(int j=0;j<Nprof;j++){
      fprintf(fh," %f",LCA[i*Nprof+j]);
    }
    fprintf(fh,"\n");
  }
  fclose(fh);

  //******************************************************************************************
  




  free(LCA);
  free(LCB);
  free(DLCA);
  free(DLCB);
  return 0;
}
