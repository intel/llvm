// REQUIRES: aspect-fp64
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// RUN: %{build} -fno-builtin -fsycl-device-lib-jit-link -o %t.out
// RUN: %{run} %t.out
//
// UNSUPPORTED: cuda || hip
// end INTEL_CUSTOMIZATION

#include "imf_utils.hpp"
#include <sycl/ext/intel/math.hpp>

int main(int, char **) {
  sycl::queue device_queue(sycl::default_selector_v);

  {
    std::initializer_list<double> input_vals1 = {
        0x1.5ef3da7bf609ap+4, 0x1.fbd37afb0f8edp-1, 0x1.9238e38e38e35p+6,
        0x1.7p+3};
    std::initializer_list<double> input_vals2 = {
        -0x1.bc7db6de6d33fp+9, 0x1.2f638fa4e71a6p+10, 0x1.08e38e38e38e3p+4,
        -0x1.94p+3};
    std::initializer_list<unsigned long long> ref_vals_rd = {
        0xc08b186180a8d83b, 0x4092fa30a14467c5, 0x405d471c71c71c6d,
        0xbff2000000000000};
    std::initializer_list<unsigned long long> ref_vals_rn = {
        0xc08b186180a8d83a, 0x4092fa30a14467c5, 0x405d471c71c71c6e,
        0xbff2000000000000};
    std::initializer_list<unsigned long long> ref_vals_ru = {
        0xc08b186180a8d83a, 0x4092fa30a14467c6, 0x405d471c71c71c6e,
        0xbff2000000000000};
    std::initializer_list<unsigned long long> ref_vals_rz = {
        0xc08b186180a8d83a, 0x4092fa30a14467c5, 0x405d471c71c71c6d,
        0xbff2000000000000};
    test2(device_queue, input_vals1, input_vals2, ref_vals_rd,
          F2T(unsigned long long, sycl::ext::intel::math::dadd_rd));
    std::cout << "sycl::ext::intel::math::dadd_rd passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals_rn,
          F2T(unsigned long long, sycl::ext::intel::math::dadd_rn));
    std::cout << "sycl::ext::intel::math::dadd_rn passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals_ru,
          F2T(unsigned long long, sycl::ext::intel::math::dadd_ru));
    std::cout << "sycl::ext::intel::math::dadd_ru passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals_rz,
          F2T(unsigned long long, sycl::ext::intel::math::dadd_rz));
    std::cout << "sycl::ext::intel::math::dadd_rz passes." << std::endl;
  }

  {
    std::initializer_list<double> input_vals1 = {
        0x1.5ef3da7bf609ap+4, 0x1.fbd37afb0f8edp-1, 0x1.9238e38e38e35p+6,
        0x1.7p+3};
    std::initializer_list<double> input_vals2 = {
        -0x1.bc7db6de6d33fp+9, 0x1.2f638fa4e71a6p+10, 0x1.08e38e38e38e3p+4,
        -0x1.94p+3};
    std::initializer_list<unsigned long long> ref_vals_rd = {
        0x408c77555b24ce43, 0xc092f24153587b87, 0x4054fffffffffffc,
        0x4038200000000000};
    std::initializer_list<unsigned long long> ref_vals_rn = {
        0x408c77555b24ce44, 0xc092f24153587b87, 0x4054fffffffffffc,
        0x4038200000000000};
    std::initializer_list<unsigned long long> ref_vals_ru = {
        0x408c77555b24ce44, 0xc092f24153587b86, 0x4054fffffffffffd,
        0x4038200000000000};
    std::initializer_list<unsigned long long> ref_vals_rz = {
        0x408c77555b24ce43, 0xc092f24153587b86, 0x4054fffffffffffc,
        0x4038200000000000};
    test2(device_queue, input_vals1, input_vals2, ref_vals_rd,
          F2T(unsigned long long, sycl::ext::intel::math::dsub_rd));
    std::cout << "sycl::ext::intel::math::dsub_rd passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals_rn,
          F2T(unsigned long long, sycl::ext::intel::math::dsub_rn));
    std::cout << "sycl::ext::intel::math::dsub_rn passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals_ru,
          F2T(unsigned long long, sycl::ext::intel::math::dsub_ru));
    std::cout << "sycl::ext::intel::math::dsub_ru passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals_rz,
          F2T(unsigned long long, sycl::ext::intel::math::dsub_rz));
    std::cout << "sycl::ext::intel::math::dsub_rz passes." << std::endl;
  }

  {
    std::initializer_list<double> input_vals1 = {
        0x1.5ef3da7bf609ap+4, 0x1.fbd37afb0f8edp-1, 0x1.9238e38e38e35p+6,
        0x1.7p+3};
    std::initializer_list<double> input_vals2 = {
        -0x1.bc7db6de6d33fp+9, 0x1.2f638fa4e71a6p+10, 0x1.08e38e38e38e3p+4,
        -0x1.94p+3};
    std::initializer_list<unsigned long long> ref_vals_rd = {
        0xc0d30ada3597be04, 0x4092cea6724fb0f1, 0x409a030329161f95,
        0xc062260000000000};
    std::initializer_list<unsigned long long> ref_vals_rn = {
        0xc0d30ada3597be04, 0x4092cea6724fb0f2, 0x409a030329161f96,
        0xc062260000000000};
    std::initializer_list<unsigned long long> ref_vals_ru = {
        0xc0d30ada3597be03, 0x4092cea6724fb0f2, 0x409a030329161f96,
        0xc062260000000000};
    std::initializer_list<unsigned long long> ref_vals_rz = {
        0xc0d30ada3597be03, 0x4092cea6724fb0f1, 0x409a030329161f95,
        0xc062260000000000};
    test2(device_queue, input_vals1, input_vals2, ref_vals_rd,
          F2T(unsigned long long, sycl::ext::intel::math::dmul_rd));
    std::cout << "sycl::ext::intel::math::dmul_rd passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals_rn,
          F2T(unsigned long long, sycl::ext::intel::math::dmul_rn));
    std::cout << "sycl::ext::intel::math::dmul_rn passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals_ru,
          F2T(unsigned long long, sycl::ext::intel::math::dmul_ru));
    std::cout << "sycl::ext::intel::math::dmul_ru passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals_rz,
          F2T(unsigned long long, sycl::ext::intel::math::dmul_rz));
    std::cout << "sycl::ext::intel::math::dmul_rz passes." << std::endl;
  }

  return 0;
}
