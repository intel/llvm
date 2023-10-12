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
    std::initializer_list<float> input_vals1 = {0x1p-1, 0x1.8bd054p+6,
                                                0x1.fcd686p+0, -0x1.7f9abp+3};
    std::initializer_list<float> input_vals2 = {-0x1.a8p+2, -0x1.674a3cp+5,
                                                0x1.f3d6aep+10, 0x1.d6bf48p+10};
    std::initializer_list<unsigned> ref_vals_rd = {0xc0c40000, 0x42582b36,
                                                   0x44fa2af1, 0x44e9e009};
    std::initializer_list<unsigned> ref_vals_rn = {0xc0c40000, 0x42582b36,
                                                   0x44fa2af2, 0x44e9e009};
    std::initializer_list<unsigned> ref_vals_ru = {0xc0c40000, 0x42582b36,
                                                   0x44fa2af2, 0x44e9e00a};
    std::initializer_list<unsigned> ref_vals_rz = {0xc0c40000, 0x42582b36,
                                                   0x44fa2af1, 0x44e9e009};
    test2(device_queue, input_vals1, input_vals2, ref_vals_rd,
          F2T(unsigned, sycl::ext::intel::math::fadd_rd));
    std::cout << "sycl::ext::intel::math::fadd_rd passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals_rn,
          F2T(unsigned, sycl::ext::intel::math::fadd_rn));
    std::cout << "sycl::ext::intel::math::fadd_rn passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals_ru,
          F2T(unsigned, sycl::ext::intel::math::fadd_ru));
    std::cout << "sycl::ext::intel::math::fadd_ru passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals_rz,
          F2T(unsigned, sycl::ext::intel::math::fadd_rz));
    std::cout << "sycl::ext::intel::math::fadd_rz passes." << std::endl;
  }

  {
    std::initializer_list<float> input_vals1 = {0x1p-1, 0x1.8bd054p+6,
                                                0x1.fcd686p+0, -0x1.7f9abp+3};
    std::initializer_list<float> input_vals2 = {-0x1.a8p+2, -0x1.674a3cp+5,
                                                0x1.f3d6aep+10, 0x1.d6bf48p+10};
    std::initializer_list<unsigned> ref_vals_rd = {0x40e40000, 0x430fdd5c,
                                                   0xc4f9abbd, 0xc4ecdf3f};
    std::initializer_list<unsigned> ref_vals_rn = {0x40e40000, 0x430fdd5c,
                                                   0xc4f9abbc, 0xc4ecdf3f};
    std::initializer_list<unsigned> ref_vals_ru = {0x40e40000, 0x430fdd5d,
                                                   0xc4f9abbc, 0xc4ecdf3e};
    std::initializer_list<unsigned> ref_vals_rz = {0x40e40000, 0x430fdd5c,
                                                   0xc4f9abbc, 0xc4ecdf3e};
    test2(device_queue, input_vals1, input_vals2, ref_vals_rd,
          F2T(unsigned, sycl::ext::intel::math::fsub_rd));
    std::cout << "sycl::ext::intel::math::fsub_rd passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals_rn,
          F2T(unsigned, sycl::ext::intel::math::fsub_rn));
    std::cout << "sycl::ext::intel::math::fsub_rn passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals_ru,
          F2T(unsigned, sycl::ext::intel::math::fsub_ru));
    std::cout << "sycl::ext::intel::math::fsub_ru passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals_rz,
          F2T(unsigned, sycl::ext::intel::math::fsub_rz));
    std::cout << "sycl::ext::intel::math::fsub_rz passes." << std::endl;
  }

  {
    std::initializer_list<float> input_vals1 = {0x1p-1, 0x1.8bd054p+6,
                                                0x1.fcd686p+0, -0x1.7f9abp+3};
    std::initializer_list<float> input_vals2 = {-0x1.a8p+2, -0x1.674a3cp+5,
                                                0x1.f3d6aep+10, 0x1.d6bf48p+10};
    std::initializer_list<unsigned> ref_vals_rd = {0xc0540000, 0xc58ae0fc,
                                                   0x45786037, 0xc6b05928};
    std::initializer_list<unsigned> ref_vals_rn = {0xc0540000, 0xc58ae0fb,
                                                   0x45786037, 0xc6b05928};
    std::initializer_list<unsigned> ref_vals_ru = {0xc0540000, 0xc58ae0fb,
                                                   0x45786038, 0xc6b05927};
    std::initializer_list<unsigned> ref_vals_rz = {0xc0540000, 0xc58ae0fb,
                                                   0x45786037, 0xc6b05927};
    test2(device_queue, input_vals1, input_vals2, ref_vals_rd,
          F2T(unsigned, sycl::ext::intel::math::fmul_rd));
    std::cout << "sycl::ext::intel::math::fmul_rd passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals_rn,
          F2T(unsigned, sycl::ext::intel::math::fmul_rn));
    std::cout << "sycl::ext::intel::math::fmul_rn passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals_ru,
          F2T(unsigned, sycl::ext::intel::math::fmul_ru));
    std::cout << "sycl::ext::intel::math::fmul_ru passes." << std::endl;
    test2(device_queue, input_vals1, input_vals2, ref_vals_rz,
          F2T(unsigned, sycl::ext::intel::math::fmul_rz));
    std::cout << "sycl::ext::intel::math::fmul_rz passes." << std::endl;
  }

  return 0;
}
