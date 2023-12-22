#include "magnification_map.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>


Mpd MagnificationMap::getFullMpd(){
  Mpd theMpd(0);
  try {
    if( this->convolved ){
      throw "Map is convolved. It has to be in ray counts in order to use this function.";
    }
    thrust::device_vector<int> counts;
    thrust::device_vector<double> bins;
    thrust::device_vector<double> data(this->data,this->data+this->Nx*this->Ny);
    thrust::sort(data.begin(),data.end());

    int num_bins = thrust::inner_product(data.begin(),data.end()-1,data.begin()+1,int(1),thrust::plus<int>(),thrust::not_equal_to<double>());
    counts.resize(num_bins);
    bins.resize(num_bins);

    thrust::reduce_by_key(data.begin(),data.end(),thrust::constant_iterator<int>(1),bins.begin(),counts.begin());
    thrust::host_vector<int> hcounts(counts);
    thrust::host_vector<double> hbins(bins);
    
    theMpd.reset(num_bins);
    for(unsigned int i=0;i<hcounts.size();i++){
      theMpd.counts[i] = (double) (hcounts[i])/(double) (this->Nx*this->Ny);
      theMpd.bins[i]   = ((double) (hbins[i]));
    }
  } catch(const char* msg){
    std::cout << msg << std::endl;
  }
  return theMpd;
}

Mpd MagnificationMap::getBinnedMpd(int Nbins,double min=0.02,double max=200){
  // creating bins which are evenly spaced in log space
  double logmin  = log10(min);
  double logmax  = log10(max);
  double logdbin = (logmax-logmin)/Nbins;
  double* bins   = (double*) calloc(Nbins,sizeof(double));
  for(int i=0;i<Nbins;i++){
    bins[i] = pow(10,logmin+(i+1)*logdbin);
  }

  thrust::device_vector<int>    counts(Nbins);
  thrust::device_vector<double> dbins(bins,bins+Nbins);
  thrust::device_vector<double> data(this->data,this->data+this->Nx*this->Ny);
  thrust::sort(data.begin(),data.end());

  // For the following lines to work I need to compile using the flag: --expt-extended-lambda
  //  auto getLog10LambdaFunctor = [=]  __device__ (double x) {return log10(x);};
  //  thrust::transform(data.begin(),data.end(),data.begin(),getLog10LambdaFunctor);

  double range[2] = {min,max};
  thrust::device_vector<double> drange(range,range+2);
  thrust::device_vector<int>    dirange(2);
  thrust::lower_bound(data.begin(),data.end(),drange.begin(),drange.end(),dirange.begin());
  thrust::host_vector<int> hirange(dirange);
  //  std::cout << hirange[0] << " " << hirange[1] << std::endl;

  thrust::upper_bound(data.begin() + hirange[0],data.begin() + hirange[1],dbins.begin(),dbins.end(),counts.begin());
  //  thrust::upper_bound(data.begin(),data.end(),dbins.begin(),dbins.end(),counts.begin());
  thrust::adjacent_difference(counts.begin(),counts.end(),counts.begin());
  thrust::host_vector<int>    hcounts(counts);

  Mpd theMpd(hcounts.size());
  for(unsigned int i=0;i<hcounts.size();i++){
    if( hcounts[i] < 1 ){
      theMpd.counts[i] = 1.0/(double) (this->Nx*this->Ny);
    } else {
      theMpd.counts[i] = (double) (hcounts[i]) /(double) (this->Nx*this->Ny);
    }
    theMpd.bins[i]   = (double) bins[i];
  }
  free(bins);
  return theMpd;
}
