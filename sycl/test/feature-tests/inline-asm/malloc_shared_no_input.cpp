// UNSUPPORTED: cuda
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -DINLINE_ASM -o %t.out
// RUN: %t.out
// RUN: %clangxx -fsycl %s -o %t.ref.out
// RUN: %t.ref.out

#include "include/asmhelper.h"
#include <CL/sycl.hpp>
#include <iostream>

constexpr size_t problem_size = 16;

class kernel_name;

int main() {
  cl::sycl::queue q;
  cl::sycl::device Device = q.get_device();

  if (!isInlineASMSupported(Device) || !Device.has_extension("cl_intel_required_subgroup_size")) {
    std::cout << "Skipping test\n";
    return 0;
  }
  auto ctx = q.get_context();
  int *a = (int *)malloc_shared(sizeof(int) * problem_size, q.get_device(), ctx);
  for (int i = 0; i < problem_size; i++)
    a[i] = i;

  q.submit([&](cl::sycl::handler &cgh) {
     cgh.parallel_for<kernel_name>(
         cl::sycl::range<1>(problem_size), [=](cl::sycl::id<1> idx) [[cl::intel_reqd_sub_group_size(16)]] {
           int i = idx[0];
#if defined(INLINE_ASM) && defined(__SYCL_DEVICE_ONLY__)
           asm volatile("mov (M1, 16) %0(0,0)<1> 0x7:d"
                        : "=rw"(a[i]));
#else
           a[i] = 7;
#endif
         });
   }).wait();

  bool currect = true;
  for (int i = 0; i < problem_size; i++) {
    if (a[i] != 7) {
      currect = false;
      std::cerr << "error in a[" << i << "]="
                << a[i] << "!=" << 7 << std::endl;
      break;
    }
  }

  if (!currect) {
    std::cerr << "Error" << std::endl;
    cl::sycl::free(a, ctx);
    return 1;
  }

  std::cerr << "Pass" << std::endl;
  cl::sycl::free(a, ctx);
  return 0;
}
