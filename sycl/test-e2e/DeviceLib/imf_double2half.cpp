// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// RUN: %clangxx -fsycl -fno-builtin -fsycl-device-lib-jit-link %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// UNSUPPORTED: cuda || hip

#include "imf_utils.hpp"

extern "C" {
_iml_half_internal __imf_double2half(double);
}

int main() {

  sycl::queue device_queue(sycl::default_selector_v);
  std::cout << "Running on "
            << device_queue.get_device().get_info<sycl::info::device::name>()
            << "\n";

  if (!device_queue.get_device().has(sycl::aspect::fp64)) {
    std::cout << "Test skipped on platform without fp64 support." << std::endl;
    return 0;
  }

  if (!device_queue.get_device().has(sycl::aspect::fp16)) {
    std::cout << "Test skipped on platform without fp16 support." << std::endl;
    return 0;
  }

  {
    std::initializer_list<uint64_t> input_vals = {
        0,                  // 0
        0x7FF0000000000000, // +infinity
        0xFFF0000000000000, // -infinity
        0x4026800000000000, // 11.25
        0x409025643C8E4F03, // 1033.3478872524
        0x40EFFC0000000000, // 65504
        0xC0EFFC0000000000, // -65504
        0xC0D38814311F5D54, // -20000.31549820055245
        0x409F9B8D12ACEFA7, // 2022.887766554
        0x40ee120000000000, // 61584
        0xC0EE160000000000, // -61616
        0x40FAA93000000000, // 109203
        0xC1A7D8B7FF20E365, // -200039423.56423487283
        0x3C370EF54646D497, // 1.25e-18
        0xBCB1B3CFC61ACF52, // -2.4567e-16
        0x39F036448D68D482, // 1.2789e-29
        0xB99C100A89BE0A2D, // -3.45899e-31
    };
    std::initializer_list<uint16_t> ref_vals = {
        0,      0x7C00, 0xFC00, 0x49A0, 0x6409, 0x7BFF, 0xFBFF, 0xF4E2, 0x67E7,
        0x7B84, 0xFB86, 0x7C00, 0xFC00, 0,      0x8000, 0,      0x8000};

    test_host(input_vals, ref_vals, F_Half4(__imf_double2half));
    test(device_queue, input_vals, ref_vals, F_Half4(__imf_double2half));
  }

  return 0;
}
