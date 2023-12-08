#include <vector>
#include <string>

#include "variability_models.hpp"


int main(int argc,char* argv[]){
  std::string map_id = "12115";
  std::vector<double> sizes{0.1,0.2,0.3};
  int Nprof = sizes.size();
  std::string shape = "gaussian";
  int Nloc = 3;
  int loc_x[Nloc] = {1000,2000,3000};
  int loc_y[Nloc] = {1000,2000,3000};


  
  double* LC = (double*) malloc(Nloc*Nprof*sizeof(double));
  expanding_source(map_id,sizes,shape,Nloc,loc_x,loc_y,LC);


  FILE* fh = fopen("lcurves.dat","w");
  for(int i=0;i<Nloc;i++){
    for(int j=0;j<Nprof;j++){
      fprintf(fh," %f",LC[i*Nprof+j]);
    }
    fprintf(fh,"\n");
  }
  fclose(fh);
  

  free(LC);
  return 0;
}
