// RUN: %clangxx -fsycl -o query %s
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

  size_t dmsize = myparams::defaultM;
  size_t dnsize = myparams::defaultN;
  size_t dksize = myparams::defaultK;
  std::cout << "sizes of AMX tpu_params chosen by the user are: M " << dmsize
            << " N " << dnsize << " K " << dksize << std::endl;

  // Sizes-only query: types are given, generate default sizes
  using myparams2 = tpu_params<tpu::amx, int8_t, int8_t, int>;
  myparams2 p;
  dmsize = myparams2::defaultM;
  dnsize = myparams2::defaultN;
  dksize = myparams2::defaultK;
  std::cout << "default AMX sizes tpu_params  are: M " << dmsize << " N "
            << dnsize << " K " << dksize << "\n AMX int8 num combinations is "
            << p.num_combinations << std::endl;

  // general query: types are not given
  tpu_params<tpu::amx> myparams3;

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
          myparams2::joint_matrix_a<sub_group> sub_a1(sg);
          myparams2::joint_matrix_b<sub_group> sub_b1(sg);
          myparams2::joint_matrix_c<sub_group> sub_c1(sg);

          joint_matrix<unsigned short, msize, ksize> sub_a(sg);
          joint_matrix<unsigned short, ksize, nsize> sub_b(sg);
          joint_matrix<float, msize, nsize> sub_c(sg);
        });
  });
}

void query_dpas() {

  // generates combination assert
  // using myparams = tpu_params<tpu::dpas, int, int, int, 2, 8, 32>;

  // generate combination of type assert
  // using myparams = tpu_params<tpu::dpas, int, int, int>;

  // tells whether a combination is valid or not, if valid, those will be set as
  // default
  using myparams = tpu_params<tpu::dpas, int8_t, int8_t, int, 2, 8, 32>;

  size_t dmsize = myparams::defaultM;
  size_t dnsize = myparams::defaultN;
  size_t dksize = myparams::defaultK;
  std::cout << "sizes of DPAS tpu_params chosen by the user are: M " << dmsize
            << " N " << dnsize << " K " << dksize << std::endl;

  // sizes-only query: types are given, generate default sizes
  using myparams2 = tpu_params<tpu::dpas, int8_t, int8_t, int>;
  myparams2 p;
  dmsize = myparams2::defaultM;
  dnsize = myparams2::defaultN;
  dksize = myparams2::defaultK;
  std::cout << "Default DPAS sizes  are: M " << dmsize << " N " << dnsize
            << " K " << dksize << "\n DPAS int8 num combinations is "
            << p.num_combinations << std::endl;

  dmsize = myparams2::combinations[0].msize;
  dnsize = myparams2::combinations[0].nsize;
  dksize = myparams2::combinations[0].ksize;
  std::cout << "one of DPAS combination sizes  is: M " << dmsize << " N "
            << dnsize << " K " << dksize << std::endl;

  // general query: types are not given
  tpu_params<tpu::dpas> myparams3;
  std::cout << "DPAS query num combinations: " << myparams3.num_combinations
            << std::endl;

  if (myparams3.combinations[0].msize == 0) // this is not a max params hardware
    return;
  constexpr int msize = myparams3.combinations[0].msize;
  constexpr int nsize = myparams3.combinations[0].nsize;
  constexpr int ksize = myparams3.combinations[0].ksize;
  std::cout << "DPAS query sizes are: M " << msize << " N " << nsize << " K "
            << ksize << std::endl;
  std::cout << "DPAS query max sizes are: M "
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
          myparams2::joint_matrix_a<sub_group> sub_a1(sg);
          myparams2::joint_matrix_b<sub_group> sub_b1(sg);
          myparams2::joint_matrix_c<sub_group> sub_c1(sg);

          joint_matrix<unsigned short, msize, ksize> sub_a(sg);
          joint_matrix<unsigned short, ksize, nsize> sub_b(sg);
          joint_matrix<float, msize, nsize> sub_c(sg);
        });
  });
}

int main() {
  query_amx();
  query_dpas();
  return 0;
}
