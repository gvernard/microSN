#include <cstdlib> // for std::size_t
#include <iostream>

#include "util.hpp"


/******************************************************************************
CUFFT error checking

******************************************************************************/
/*
bool cufft_error(cufftResult result, const char* name, const char* file, const int line){
  if( result != CUFFT_SUCCESS ){
    std::cerr << "CUFFT error check for: " << name << " failed at " << file << ":" << line << "\n";
    std::cerr << "Error code: " << result << "\n";
    return true;
  } else {
    return false;
  }
}
*/


/******************************************************************************
CUDA error checking

\param name -- to print in error msg
\param sync -- boolean of whether device needs synchronized or not
\param file -- the file being run
\param line -- line number of the source code where the error is given

\return bool -- true for error, false for no error
******************************************************************************/
bool cuda_error(const char* name, bool sync, const char* file, const int line)
{
    cudaError_t err = cudaGetLastError();

    /******************************************************************************
    if the last error message is not a success, print the error code and msg and
    return true (i.e., an error occurred)
    ******************************************************************************/
    if (err != cudaSuccess)
    {
        const char* errMsg = cudaGetErrorString(err);
        std::cerr << "CUDA error check for " << name << " failed at " << file << ":" << line << "\n";
        std::cerr << "Error code: " << err << " (" << errMsg << ")\n";
        return true;
    }

    /******************************************************************************
    if a device synchronization is also to be done
    ******************************************************************************/
    if (sync)
    {
        /******************************************************************************
        perform the same error checking as initially
        ******************************************************************************/
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            const char* errMsg = cudaGetErrorString(err);
            std::cerr << "CUDA error check for cudaDeviceSynchronize failed at " << file << ":" << line << "\n";
            std::cerr << "Error code: " << err << " (" << errMsg << ")\n";
            return true;
        }
    }
    return false;
}

/******************************************************************************
Normal error checking

\param name -- to print in error msg
\param file -- the file being run
\param line -- line number of the source code where the error is given

\return bool -- always true for error
******************************************************************************/
bool normal_error(const char* name, const char* file, const int line){
  std::cerr << "Error: '" << name << "' failed at " << file << ":" << line << "\n";
  return true;
}


/******************************************************************************
set the number of threads per block for the kernel

\param threads -- reference to threads
\param x -- number of threads per block in x dimension
\param y -- number of threads per block in y dimension
\param z -- number of threads per block in z dimension
******************************************************************************/
/*
void set_threads(dim3& threads, int x = 1, int y = 1, int z = 1)
{
    threads.x = x;
    threads.y = y;
    threads.z = z;
}
*/

/******************************************************************************
set the number of blocks for the kernel

\param threads -- reference to threads
\param blocks -- reference to blocks
\param x -- number of threads in x dimension
\param y -- number of threads in y dimension
\param z -- number of threads in z dimension
******************************************************************************/
/*
void set_blocks(dim3& threads, dim3& blocks, int x = 1, int y = 1, int z = 1)
{
    blocks.x = (x - 1) / threads.x + 1;
    blocks.y = (y - 1) / threads.y + 1;
    blocks.z = (z - 1) / threads.z + 1;
}
*/

/******************************************************************************
get the amount of available memory, in bytes, on the gpu

\return free -- std::size_t of the number of available bytes free
******************************************************************************/
std::size_t get_free_gpu_mem()
{
    std::size_t free = 0;
    std::size_t total = 0;

    cudaMemGetInfo (&free, &total);

    return free/(1024*1024);
}


/******************************************************************************
get the amount of total memory, in bytes, on the gpu

\return total -- std::size_t of the number of available bytes free
******************************************************************************/
std::size_t get_total_gpu_mem()
{
    std::size_t free = 0;
    std::size_t total = 0;

    cudaMemGetInfo (&free, &total);

    return total/(1024*1024);
}

/******************************************************************************
get the amount of used memory, in bytes, on the gpu

\return total -- std::size_t of the number of available bytes free
******************************************************************************/
std::size_t get_used_gpu_mem()
{
    std::size_t free = 0;
    std::size_t total = 0;

    cudaMemGetInfo (&free, &total);

    return (total-free)/(1024*1024);
}