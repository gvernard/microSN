#ifndef DATA_STRUCTS_HPP
#define DATA_STRUCTS_HPP

class Chi2Vars {
public:
  int Njp;
  int* indA;      // size of Nj
  int* indB;      // size of Nj
  double* facA;   // size of Nj
  double* facB;   // size of Nj
  double* new_d;  // size of Nj
  double* new_s;  // size of Nj

  Chi2Vars(int Njp);
  Chi2Vars(const Chi2Vars& other);
  ~Chi2Vars();
};


class Chi2 {
public:
  int Nloc;
  double* values;    // size of Nloc*Nloc on the CPU
  double* d_values;  // size of Nloc*Nloc on the GPU

  Chi2(int Nloc);
  Chi2(const Chi2& other); // The copy constructor is not expected to be used
  ~Chi2();
  void transfer_to_CPU();
};


class Chi2SortBins {
public:
  int Nloc;
  int Nbins;
  unsigned int* sorted_ind;    // size of Nloc*Nloc on the CPU
  unsigned int* d_sorted_ind;  // size of Nloc*Nloc on the GPU
  unsigned int* lower_ind;     // size of Nbins on the CPU
  unsigned int* n_per_bin;     // size of Nbins on the CPU
  
  Chi2SortBins(int Nloc,int Nbins);
  Chi2SortBins(const Chi2SortBins& other); // The copy constructor is not expected to be used
  ~Chi2SortBins();
};


class SimLC {
public:
  int Nloc;
  int Nprof;
  double* LC;     // size of Nloc*Nprof on the CPU
  double* DLC;    // size of Nloc*(Nprof-1) on the CPU
  double* d_LC;   // size of Nloc*Nprof on the GPU
  double* d_DLC;  // size of Nloc*(Nprof-1) on the GPU

  SimLC(int Nloc,int Nprof);
  SimLC(const SimLC& other); // The copy constructor is not expected to be used
  ~SimLC();
  void transfer_to_CPU();
};

#endif /* DATA_STRUCTS_HPP */
