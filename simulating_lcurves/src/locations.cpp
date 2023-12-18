#include <cmath>

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


void locations_random(int Nloc,int* x,int* y){

}
