#ifndef DATA_STRUCTS_HPP
#define DATA_STRUCTS_HPP

class Chi2Vars {
public:
  int Njp;
  int* indA;
  int* indB;
  double* facA;
  double* facB;
  double* new_d;
  double* new_s;

  Chi2Vars(int Njp);
  Chi2Vars(const Chi2Vars& other);
  ~Chi2Vars();
};


class Chi2 {
public:
  int Nloc;
  double* values;
  double* d_values;

  Chi2(int Nloc);
  Chi2(const Chi2& other); // The copy constructor is not expected to be used
  ~Chi2();
  void transfer_to_CPU();
};


class Chi2SortBins {
public:
  int Nloc;
  int Nbins;
  unsigned int* sorted_ind;
  unsigned int* d_sorted_ind;
  unsigned int* upper_ind;
  unsigned int* n_per_bin;
  
  Chi2SortBins(int Nloc,int Nbins);
  Chi2SortBins(const Chi2SortBins& other); // The copy constructor is not expected to be used
  ~Chi2SortBins();
};


class SimLC {
public:
  int Nloc;
  int Nprof;
  double* LC;
  double* DLC;
  double* d_LC;
  double* d_DLC;

  SimLC(int Nloc,int Nprof);
  SimLC(const SimLC& other); // The copy constructor is not expected to be used
  ~SimLC();
  void transfer_to_CPU();
};

#endif /* DATA_STRUCTS_HPP */
