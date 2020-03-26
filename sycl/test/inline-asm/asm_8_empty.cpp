// UNSUPPORTED: cuda
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -DINLINE_ASM -o %t.out
// RUN: %t.out

#include "include/asmcheck.h"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

constexpr int LIST_SIZE = 1024;
using arr_t = std::vector<cl::sycl::cl_int>;
constexpr auto sycl_write = cl::sycl::access::mode::write;

// class is used for kernel name
template <typename T>
class no_opts;

template <typename T>
void process_buffers(cl::sycl::queue &deviceQueue, T *pc, size_t sz) {
  cl::sycl::range<1> numOfItems{sz};
  cl::sycl::buffer<T, 1> bufferC(pc, numOfItems);

  deviceQueue.submit([&](cl::sycl::handler &cgh) {
    auto C = bufferC.template get_access<sycl_write>(cgh);

    auto kern = [C](cl::sycl::id<1> wiID)
        [[cl::intel_reqd_sub_group_size(8)]] {
      C[wiID] = 43;
#if defined(INLINE_ASM) && defined(__SYCL_DEVICE_ONLY__)
      asm volatile("");
#endif
    };
    cgh.parallel_for<class no_opts<T>>(numOfItems, kern);
  });
};

int main() {
  arr_t C(LIST_SIZE);

  cl::sycl::gpu_selector gpsel;
  cl::sycl::queue deviceQueue(gpsel);

  sycl::device Device = deviceQueue.get_device();

  if (!isInlineASMSupported(Device) || !Device.has_extension("cl_intel_required_subgroup_size")) {
    std::cout << "Skipping test\n";
    return 0;
  }

  for (int i = 0; i < LIST_SIZE; i++) {
    C[i] = 0;
  }

  process_buffers(deviceQueue, C.data(), LIST_SIZE);

  bool all_right = true;

  for (int i = 0; i < LIST_SIZE; ++i)
    if (C[i] != 43) {
      std::cerr << "At index: " << i << ". ";
      std::cerr << C[i] << " != " << 43 << "\n";
      all_right = false;
      break;
    }
  if (all_right) {
    std::cout << "Pass" << std::endl;
    return 0;
  }
  std::cout << "Error" << std::endl;
  return -1;
}
