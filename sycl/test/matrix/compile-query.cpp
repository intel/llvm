// RUN: %clangxx -fsycl -o compile-query %s
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::oneapi::experimental::matrix;

void query_amx_spr() {

  // generates combination assert
  // using myparams = matrix_params<architecture::intel_cpu_spr, int, int, int,
  // 2, 8, 32>;

  // generates types assert
  // using myparams2 = matrix_params<architecture::intel_cpu_spr, int, int,
  // int>;

  // tells whether a combination is valid or not, if valid, those will be set as
  // default
  using myparams = matrix_params<architecture::intel_cpu_spr, int8_t, int8_t,
                                 int, int, 2, 8, 32>;

  size_t dmsize = myparams::M;
  size_t dnsize = myparams::N;
  size_t dksize = myparams::K;
  std::cout << "sizes of AMX matrix_params chosen by the user are: M " << dmsize
            << " N " << dnsize << " K " << dksize << std::endl;

  // Sizes-only query: types are given, generate default sizes
  using myparams2 =
      matrix_params<architecture::intel_cpu_spr, int8_t, int8_t, int>;
  myparams2 p;
  dmsize = myparams2::M;
  dnsize = myparams2::N;
  dksize = myparams2::K;
  std::cout << "default AMX sizes matrix_params  are: M " << dmsize << " N "
            << dnsize << " K " << dksize << std::endl;
  return;
}

void query_xmx_dg2() {

  // generates combination assert
  // using myparams = matrix_params<architecture::intel_gpu_dg2_g10, int, int,
  // int, 2, 8, 32>;

  // generate combination of type assert
  // using myparams = matrix_params<architecture::intel_gpu_dg2_g10, int, int,
  // int>;

  // tells whether a combination is valid or not, if valid, those will be set as
  // default
  using myparams = matrix_params<architecture::intel_gpu_dg2_g10, int8_t,
                                 int8_t, int, int, 2, 8, 32>;

  size_t dmsize = myparams::M;
  size_t dnsize = myparams::N;
  size_t dksize = myparams::K;
  std::cout << "sizes of Intel XMX of architecture::intel_gpu_dg2_g10 "
               "matrix_params chosen by the user are: M "
            << dmsize << " N " << dnsize << " K " << dksize << std::endl;

  // sizes-only query: types are given, generate default sizes
  using myparams2 =
      matrix_params<architecture::intel_gpu_dg2_g10, int8_t, int8_t, int>;
  dmsize = myparams2::M;
  dnsize = myparams2::N;
  dksize = myparams2::K;
  std::cout
      << "Default Intel XMX of architecture::intel_gpu_dg2_g10 sizes  are: M "
      << dmsize << " N " << dnsize << " K " << dksize << std::endl;
  return;
}

void query_xmx_pvc() {

  // generates combination assert
  // using myparams = matrix_params<architecture::intel_gpu_pvc, int, int, int,
  // 2, 8, 32>;

  // generate combination of type assert
  // using myparams = matrix_params<architecture::intel_gpu_pvc, int, int, int>;

  // tells whether a combination is valid or not, if valid, those will be set as
  // default
  using myparams = matrix_params<architecture::intel_gpu_pvc, int8_t, int8_t,
                                 int, int, 2, 16, 32>;

  size_t dmsize = myparams::M;
  size_t dnsize = myparams::N;
  size_t dksize = myparams::K;
  std::cout << "sizes of architecture::intel_gpu_pvc matrix_params chosen by "
               "the user are: M "
            << dmsize << " N " << dnsize << " K " << dksize << std::endl;

  // sizes-only query: types are given, generate default sizes
  using myparams2 =
      matrix_params<architecture::intel_gpu_pvc, int8_t, int8_t, int>;
  dmsize = myparams2::M;
  dnsize = myparams2::N;
  dksize = myparams2::K;
  std::cout << "Default Intel XMX of architecture::intel_gpu_pvc sizes  are: M "
            << dmsize << " N " << dnsize << " K " << dksize << std::endl;
  return;
}

int main() {
  query_amx_spr();
  query_xmx_dg2();
  query_xmx_pvc();
  return 0;
}
