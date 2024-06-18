// REQUIRES: aspect-fp64
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// RUN: %{build} -fno-builtin -fsycl-device-lib-jit-link -o %t.out
// RUN: %{run} %t.out
//
// UNSUPPORTED: cuda || hip

#include "imf_utils.hpp"
#include <sycl/ext/intel/math.hpp>

int main(int, char **) {
  sycl::queue device_queue(sycl::default_selector_v);
  std::initializer_list<double> input_vals = {3.0, -7.0 / 2.0, 14.0 / 15.0};
  std::initializer_list<unsigned long long> ref_vals = {
      0x3fd5555500000000, 0xbfd2492400000000, 0x3ff1249200000000};

  test(device_queue, input_vals, ref_vals,
       FT(unsigned long long, sycl::ext::intel::math::rcp64h));
  std::cout << "sycl::ext::intel::math::rcp64h passes." << std::endl;
}
