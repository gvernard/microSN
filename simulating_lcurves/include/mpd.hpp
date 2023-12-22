#ifndef MPD_HPP
#define MPD_HPP

#include <cstdlib>
#include <string>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>

class Mpd {
public:
  int Nbins;
  double* bins;
  double* counts;
  
  Mpd(int Nbins){
    this->Nbins  = Nbins;
    this->bins   = (double*) calloc(Nbins,sizeof(double));
    this->counts = (double*) calloc(Nbins,sizeof(double));
  }
  Mpd(int Nbins,double min,double max,std::string scale);
  Mpd(const Mpd& other){
    this->Nbins = other.Nbins;
    this->bins   = (double*) calloc(Nbins,sizeof(double));
    this->counts = (double*) calloc(Nbins,sizeof(double));
    for(int i=0;i<Nbins;i++){
      this->bins[i] = other.bins[i];
      this->counts[i] = other.counts[i];
    }
  }
  ~Mpd(){
    free(bins);
    free(counts);
  }
  
  void reset(int Nbins);
  void writeMpd(const std::string filename);
  double integrate();
  double sum();
  void normalize();
  void divide_by(Mpd* denominator,Mpd* ratio);
};




#endif /* MPD_HPP */
