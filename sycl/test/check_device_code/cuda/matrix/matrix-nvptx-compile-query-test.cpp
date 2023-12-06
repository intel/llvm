// REQUIRES: cuda
// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda %s -o compile-query-cuda

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::oneapi::experimental::matrix;

int main() {
  // Compile-time query to validate the matrix parameters
  using myparams = matrix_params<architecture::nvidia_gpu_sm_70, half, half,
                                 float, float, 32, 8, 16>;

  size_t dmsize = myparams::M;
  size_t dnsize = myparams::N;
  size_t dksize = myparams::K;
  std::cout
      << "sizes of Nvidia gpu sm70 matrix_params chosen by the user are: M "
      << dmsize << " N " << dnsize << " K " << dksize << std::endl;

  // Sizes-only compile-time query: types are given, generate default sizes
  using myparams2 =
      matrix_params<architecture::nvidia_gpu_sm_70, half, half, float, float>;
  myparams2 p;
  dmsize = myparams2::M;
  dnsize = myparams2::N;
  dksize = myparams2::K;
  std::cout << "default Nvidia gpu sm70 sizes matrix_params are: M " << dmsize
            << " N " << dnsize << " K " << dksize << std::endl;
  return 0;
}
