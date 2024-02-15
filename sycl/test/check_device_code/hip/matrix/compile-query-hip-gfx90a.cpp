// REQUIRES: hip
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx90a %s -o compile-query-hip

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::oneapi::experimental::matrix;

int main() {
  // Compile-time query to validate the matrix parameters
  using myparams = matrix_params<architecture::amd_gpu_gfx90a, int8_t, int8_t,
                                 int32_t, int32_t, 32, 32, 8>;

  size_t dmsize = myparams::M;
  size_t dnsize = myparams::N;
  size_t dksize = myparams::K;
  std::cout
      << "sizes of AMD gpu gfx90a matrix_params chosen by the user are: M "
      << dmsize << " N " << dnsize << " K " << dksize << std::endl;

  // Sizes-only compile-time query: types are given, generate default sizes
  using myparams2 = matrix_params<architecture::amd_gpu_gfx90a, int8_t, int8_t,
                                  int32_t, int32_t>;
  myparams2 p;
  dmsize = myparams2::M;
  dnsize = myparams2::N;
  dksize = myparams2::K;
  std::cout << "default AMD gpu gfx90a sizes matrix_params  are: M " << dmsize
            << " N " << dnsize << " K " << dksize << std::endl;
  return 0;
};
