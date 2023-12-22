#include "mpd.hpp"
#include <iostream>

Mpd::Mpd(int Nbins,double min,double max,std::string scale):Nbins(Nbins){
  this->bins = (double*) calloc(Nbins,sizeof(double));
  this->counts = (double*) calloc(Nbins,sizeof(double));

  if( scale == "log" ){
    double logmin  = log10(min);
    double logmax  = log10(max);
    double logdbin = (logmax-logmin)/Nbins;
    for(int i=0;i<Nbins;i++){
      this->bins[i] = pow(10,logmin+(i+1)*logdbin);
    }
  } else {
    double dbin = (max-min)/Nbins;
    for(int i=0;i<Nbins;i++){
      this->bins[i] = min+(i+1)*dbin;
    }
  }  
}

void Mpd::reset(int Nbins){
  free(this->bins);
  free(this->counts);
  this->Nbins  = Nbins;
  this->bins   = (double*) calloc(Nbins,sizeof(double));
  this->counts = (double*) calloc(Nbins,sizeof(double));
}

void Mpd::writeMpd(const std::string filename){
  /*
  std::ofstream output(file,std::ios::out);
  for(int i=0;i<this->Nbins;i++){
    output << std::setw(10) << std::left << this->bins[i] << std::setw(10) << std::left << this->counts[i] << std::endl;
  }
  output.close();
  */

  FILE* fh = fopen(filename.data(),"w");
  for(int i=0;i<this->Nbins;i++){
    fprintf(fh,"%11.6e %11.6e\n",this->bins[i],this->counts[i]);
  }
  fclose(fh);
}

double Mpd::integrate(){
  double sum = 0.0;
  for(int i=1;i<this->Nbins;i++){
    sum += (this->counts[i] + this->counts[i-1])*(this->bins[i] - this->bins[i-1])/2.0;
  }
  return sum;
}

double Mpd::sum(){
  double sum = 0.0;
  for(int i=0;i<this->Nbins;i++){
    sum += this->counts[i];
  }
  return sum;
}

void Mpd::normalize(){
  double norm = this->integrate();
  for(int i=0;i<this->Nbins;i++){
    this->counts[i] /= norm;
  }  
}


void Mpd::divide_by(Mpd* B,Mpd* ratio){
  int N = this->Nbins;
  double pA;
  double* int_terms = (double*) malloc(B->Nbins*sizeof(double));

  for(int i=0;i<ratio->Nbins;i++){
    double z = ratio->bins[i];
    
    for(int b=0;b<B->Nbins;b++){
      double muB = B->bins[b];
      double prod = z*muB;
     
      if( prod <= this->bins[0] ){
	pA = 1e-8;
      } else if( prod >= this->bins[this->Nbins-1] ){
	pA = 1e-8;
      } else {
	for(int k=0;k<this->Nbins;k++){
	  if( this->bins[k] > prod ){
	    pA = this->counts[k-1] + (this->counts[k] - this->counts[k-1])*(prod - this->bins[k-1])/(this->bins[k] - this->bins[k-1]);
	    break;
	  }
	}
      }
      
      int_terms[b] = muB*pA*B->counts[b];
    }

    double sum = 0.0;
    for(int b=1;b<B->Nbins;b++){
      sum += (int_terms[b-1] + int_terms[b])*(B->bins[b]-B->bins[b-1])/2.0;
    }
    ratio->counts[i] = sum;
  }
  
  free(int_terms);
}

