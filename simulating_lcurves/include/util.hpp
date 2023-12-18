#ifndef UTIL_HPP
#define UTIL_HPP

#include <cstdlib> // for std::size_t
//#include "cufft.h"

//bool cufft_error(cufftResult result, const char* name, const char* file, const int line);
bool cuda_error(const char* name, bool sync, const char* file, const int line);
bool normal_error(const char* name, const char* file, const int line);
//void set_threads(dim3& threads, int x, int y, int z);
//void set_blocks(dim3& threads, dim3& blocks, int x, int y, int z);
std::size_t get_total_gpu_mem();
std::size_t get_free_gpu_mem();
std::size_t get_used_gpu_mem();

#endif /* UTIL_HPP */
