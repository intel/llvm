// UNSUPPORTED: cuda
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -DINLINE_ASM -o %t.out
// RUN: %t.out

#include "include/asmcheck.h"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

constexpr int LIST_SIZE = 8;
using arr_t = std::vector<cl::sycl::cl_int>;
constexpr auto sycl_read = cl::sycl::access::mode::read;
constexpr auto sycl_write = cl::sycl::access::mode::write;

// class is used for kernel name
template <typename T>
class asm_no_output;

template <typename T>
void process_buffers(cl::sycl::queue &deviceQueue, T *pc, size_t sz) {
  cl::sycl::range<1> numOfItems{sz};
  cl::sycl::buffer<T, 1> bufferC(pc, numOfItems);

  deviceQueue.submit([&](cl::sycl::handler &cgh) {
    auto C = bufferC.template get_access<sycl_read>(cgh);

    auto kern = [C]()
        [[cl::intel_reqd_sub_group_size(16)]] {
      volatile int local_var = 47;
      local_var += C[0];
#if defined(INLINE_ASM) && defined(__SYCL_DEVICE_ONLY__)
      asm volatile("{\n"
                   ".decl temp v_type=G type=w num_elts=8 align=GRF\n"
                   "mov (M1,16) temp(0, 0)<1> %0(0,0)<1;1,0>\n"
                   "}\n" ::"rw"(local_var));
#else
      volatile int temp = 0;
      temp = local_var;
#endif
    };
    cgh.single_task<class asm_no_output<T>>(kern);
  });
};

int main() {
  arr_t C(LIST_SIZE);

  cl::sycl::gpu_selector gpsel;
  cl::sycl::queue deviceQueue(gpsel);
  sycl::device Device = deviceQueue.get_device();

  if (!isInlineASMSupported(Device)) {
    std::cout << "Skipping test\n";
    return 0;
  }

  for (int i = 0; i < LIST_SIZE; i++) {
    C[i] = 0;
  }

  process_buffers(deviceQueue, C.data(), LIST_SIZE);

  bool all_right = true;
  for (int i = 0; i < LIST_SIZE; ++i)
    if (C[i] != 0) {
      std::cerr << "At index: " << i << ". ";
      std::cerr << C[i] << " != " << 0 << "\n";
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
