// RUN: %clangxx -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 -fsycl -o query-use %s
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

void query_amx() {

  // generates combination assert
  // using myparams = tpu_params<tpu::amx, int, int, int, 2, 8, 32>;

  // generates types assert
  // using myparams2 = tpu_params<tpu::amx, int, int, int>;

  // tells whether a combination is valid or not, if valid, those will be set as
  // default
  using myparams = tpu_params<tpu::amx, int8_t, int8_t, int, 2, 8, 32>;

  size_t dmsize = myparams::M;
  size_t dnsize = myparams::N;
  size_t dksize = myparams::K;
  std::cout << "sizes of AMX tpu_params chosen by the user are: M " << dmsize
            << " N " << dnsize << " K " << dksize << std::endl;

  // Sizes-only query: types are given, generate default sizes
  using myparams2 = tpu_params<tpu::amx, int8_t, int8_t, int>;
  myparams2 p;
  dmsize = myparams2::M;
  dnsize = myparams2::N;
  dksize = myparams2::K;
  std::cout << "default AMX sizes tpu_params  are: M " << dmsize << " N "
            << dnsize << " K " << dksize << "\n AMX int8 num combinations is "
            << p.num_combinations << std::endl;

  // general query: types are not given
  tpu_params<tpu::amx> myparams3;

  if (myparams3.num_scopes > 0)
    if (myparams3.scopes[0] == scope_t::sub_group)
      std::cout << "There are " << myparams3.num_scopes
                << " Scopes that are supported by AMX implementation and "
                   "subgroup is one of them "
                << std::endl;

  std::cout << "AMX query num combinations: " << myparams3.num_combinations
            << std::endl;

  if (myparams3.combinations[0].msize != 0) // this is a max params hardware
    return;
  constexpr int msize = myparams3.combinations[0].max_msize;
  constexpr int nsize = myparams3.combinations[0].max_nsize;
  constexpr int ksize = myparams3.combinations[0].max_ksize;
  std::cout << "AMX query sizes are: M " << msize << " N " << nsize << " K "
            << ksize << std::endl;

  size_t NDRangeM = 1024 / msize;
  size_t NDRangeN = 1024 / nsize;
  queue q;
  q.submit([&](handler &cgh) {
    cgh.parallel_for<class imatrix>(
        nd_range<2>({NDRangeM, NDRangeN}, {1, 1}),
        [msize, ksize, nsize](nd_item<2> spmd_item) {
          sub_group sg = spmd_item.get_sub_group();
          myparams2::joint_matrix_a<sub_group, layout::row_major> sub_a1;
          myparams2::joint_matrix_b<
              sub_group, sycl::ext::intel::experimental::matrix::layout::packed>
              sub_b1;
          myparams2::joint_matrix_accumulator<sub_group> sub_c1;

          joint_matrix<sub_group, unsigned short, use::a, msize, ksize> sub_a;
          joint_matrix<sub_group, unsigned short, use::b, ksize, nsize> sub_b;
          joint_matrix<sub_group, float, use::accumulator, msize, nsize> sub_c;
        });
  });
}

void query_xmx8() {

  // generates combination assert
  // using myparams = tpu_params<tpu::xmx8, int, int, int, 2, 8, 32>;

  // generate combination of type assert
  // using myparams = tpu_params<tpu::xmx8, int, int, int>;

  // tells whether a combination is valid or not, if valid, those will be set as
  // default
  using myparams = tpu_params<tpu::xmx8, int8_t, int8_t, int, 2, 8, 32>;

  size_t dmsize = myparams::M;
  size_t dnsize = myparams::N;
  size_t dksize = myparams::K;
  std::cout << "sizes of XMX8 tpu_params chosen by the user are: M " << dmsize
            << " N " << dnsize << " K " << dksize << std::endl;

  // sizes-only query: types are given, generate default sizes
  using myparams2 = tpu_params<tpu::xmx8, int8_t, int8_t, int>;
  myparams2 p;
  dmsize = myparams2::M;
  dnsize = myparams2::N;
  dksize = myparams2::K;
  std::cout << "Default XMX8 sizes  are: M " << dmsize << " N " << dnsize
            << " K " << dksize << "\n XMX8 int8 num combinations is "
            << p.num_combinations << std::endl;

  dmsize = myparams2::combinations[0].msize;
  dnsize = myparams2::combinations[0].nsize;
  dksize = myparams2::combinations[0].ksize;
  std::cout << "one of XMX8 combination sizes  is: M " << dmsize << " N "
            << dnsize << " K " << dksize << std::endl;

  // general query: types are not given
  tpu_params<tpu::xmx8> myparams3;

  if (myparams3.num_scopes > 0)
    if (myparams3.scopes[0] == scope_t::sub_group)
      std::cout << "There are " << myparams3.num_scopes
                << " Scopes that are supported by XMX8 implementation and "
                   "subgroup is one of them "
                << std::endl;

  std::cout << "XMX8 query num combinations: " << myparams3.num_combinations
            << std::endl;

  if (myparams3.combinations[0].msize == 0) // this is not a max params hardware
    return;
  constexpr int msize = myparams3.combinations[0].msize;
  constexpr int nsize = myparams3.combinations[0].nsize;
  constexpr int ksize = myparams3.combinations[0].ksize;
  std::cout << "XMX8 query sizes are: M " << msize << " N " << nsize << " K "
            << ksize << std::endl;
  std::cout << "XMX8 query max sizes are: M "
            << myparams3.combinations[0].max_msize << " N "
            << myparams3.combinations[0].max_nsize << " K "
            << myparams3.combinations[0].max_ksize << std::endl;

  size_t NDRangeM = 1024 / msize;
  size_t NDRangeN = 1024 / nsize;
  queue q;
  q.submit([&](handler &cgh) {
    cgh.parallel_for<class dmatrix>(
        nd_range<2>({NDRangeM, NDRangeN}, {1, 1}),
        [msize, ksize, nsize](nd_item<2> spmd_item) {
          sub_group sg = spmd_item.get_sub_group();
          myparams2::joint_matrix_a<sub_group, layout::row_major> sub_a1;
          myparams2::joint_matrix_b<
              sub_group, sycl::ext::intel::experimental::matrix::layout::packed>
              sub_b1;
          myparams2::joint_matrix_accumulator<sub_group> sub_c1;

          joint_matrix<sub_group, unsigned short, use::a, msize, ksize> sub_a;
          joint_matrix<sub_group, unsigned short, use::b, ksize, nsize> sub_b;
          joint_matrix<sub_group, float, use::accumulator, msize, nsize> sub_c;
        });
  });
}

int main() {
  query_amx();
  query_xmx8();
  return 0;
}
