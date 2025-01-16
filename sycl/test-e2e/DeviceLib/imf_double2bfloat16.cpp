// REQUIRES: gpu
// REQUIRES: aspect-fp64
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// RUN: %{build} -fno-builtin -fsycl-device-lib-jit-link -o %t1.out
// RUN: %{run} %t1.out
//
// UNSUPPORTED: nvptx64-nvidia-cuda || amdgcn-amd-amdhsa

#include "imf_utils.hpp"
#include <sycl/ext/intel/math.hpp>

int main() {

  sycl::queue device_queue(sycl::default_selector_v);
  std::cout << "Running on "
            << device_queue.get_device().get_info<sycl::info::device::name>()
            << "\n";

  {
    std::initializer_list<double> input_vals = {
        __builtin_bit_cast(double, 0ULL),               // 0
        __builtin_bit_cast(double, 0x7FF0000000000000), // +infinity
        __builtin_bit_cast(double, 0xFFF0000000000000), // -infinity
        __builtin_bit_cast(double, 0x4026800000000000), // 11.25
        __builtin_bit_cast(double, 0x409025643C8E4F03), // 1033.3478872524
        __builtin_bit_cast(double, 0x40EFFC0000000000), // 65504
        __builtin_bit_cast(double, 0xC0EFFC0000000000), // -65504
        __builtin_bit_cast(double, 0xC0D38814311F5D54), // -20000.31549820055245
        __builtin_bit_cast(double, 0x409F9B8D12ACEFA7), // 2022.887766554
        __builtin_bit_cast(double, 0x40ee120000000000), // 61584
        __builtin_bit_cast(double, 0xC0EE160000000000), // -61616
        __builtin_bit_cast(double, 0x40FAA93000000000), // 109203
        __builtin_bit_cast(double, 0xC1A7D8B7FF20E365), // -200039423.564234872
        __builtin_bit_cast(double, 0x3C370EF54646D497), // 1.25e-18
        __builtin_bit_cast(double, 0xBCB1B3CFC61ACF52), // -2.4567e-16
        __builtin_bit_cast(double, 0x39F036448D68D482), // 1.2789e-29
        __builtin_bit_cast(double, 0xB99C100A89BE0A2D), // -3.45899e-31
        __builtin_bit_cast(double, 0x47EFFFFFFFFFFFFF),
        __builtin_bit_cast(double, 0x47EFF00000000000),
        __builtin_bit_cast(double, 0x47EFD00000000000),
        __builtin_bit_cast(double, 0xC7EFFFFFFFFFFFFF),
        __builtin_bit_cast(double, 0xC7EFF00000000000),
        __builtin_bit_cast(double, 0xC7EFD00000000000),
        __builtin_bit_cast(double, 0x37AFFFFFFFFFFFFF),
        __builtin_bit_cast(double, 0x380FFFFFFFFFFFFF),
    };
    std::initializer_list<uint16_t> ref_vals = {
        0x0,    0x7F80, 0xFF80, 0x4134, 0x4481, 0x4780, 0xC780, 0xC69C, 0x44FD,
        0x4771, 0xC771, 0x47D5, 0xCD3F, 0x21B8, 0xA58E, 0x0F82, 0x8CE1, 0x7F80,
        0x7F80, 0x7F7E, 0xFF80, 0xFF80, 0xFF7E, 0x2,    0x80};
    test_host(input_vals, ref_vals,
              FT(uint16_t, sycl::ext::intel::math::double2bfloat16));
    test(device_queue, input_vals, ref_vals,
         FT(uint16_t, sycl::ext::intel::math::double2bfloat16));
  }

  return 0;
}
