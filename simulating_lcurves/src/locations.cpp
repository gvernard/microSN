#include <cmath>
#include <random>

#include "locations.hpp"

void locations_on_grid(int Nside,int Nmap,int Noff,int* x,int* y){
  double dd = (Nmap - 2*Noff)/Nside;
  for(int i=0;i<Nside;i++){
    for(int j=0;j<Nside;j++){
      x[i*Nside+j] = Noff + ((int) floor(dd/2.0 + j*dd));
      y[i*Nside+j] = Noff + ((int) floor(dd/2.0 + i*dd));
    }
  }      
}


void locations_random(int Nloc,int Nmap,int Noff,int seed,int* x,int* y){
  int Neff = Nmap - 2*Noff;
  std::mt19937 gen(seed);
  std::uniform_int_distribution<unsigned> distrib(0,Neff);
  for(int i=0;i<Nloc;i++){
    x[i] = Noff + distrib(gen);
    y[i] = Noff + distrib(gen);
  }
}


