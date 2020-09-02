// UNSUPPORTED: cuda
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -DINLINE_ASM -o %t.out
// RUN: %t.out

#include "include/asmhelper.h"
#include <CL/sycl.hpp>
#include <iostream>

constexpr size_t problem_size = 100;

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
  int *b = (int *)malloc_shared(sizeof(int) * problem_size, q.get_device(), ctx);
  for (int i = 0; i < problem_size; i++) {
    b[i] = -1;
    a[i] = i;
  }

  q.submit([&](cl::sycl::handler &cgh) {
     cgh.parallel_for<kernel_name>(
         // clang-format off
         cl::sycl::range<1>(problem_size),
     [=](cl::sycl::id<1> idx) [[intel::reqd_sub_group_size(16)]] {
           // clang-format on
           int i = idx[0];
           volatile int tmp = a[i];
           tmp += 1;
#if defined(INLINE_ASM) && defined(__SYCL_DEVICE_ONLY__)
           asm volatile(" add (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<1;1,0>"
                        : "+rw"(b[i])
                        : "rw"(tmp));
#else
           b[i] += tmp;
#endif
         });
   }).wait();

  bool currect = true;
  for (int i = 0; i < problem_size; i++) {
    if (b[i] != a[i]) {
      currect = false;
      std::cerr << "error in a[" << i << "]="
                << b[i] << "!=" << a[i] << std::endl;
      break;
    }
  }

  if (!currect) {
    std::cerr << "Error" << std::endl;
    cl::sycl::free(a, ctx);
    cl::sycl::free(b, ctx);
    return 1;
  }

  std::cerr << "Pass" << std::endl;
  cl::sycl::free(a, ctx);
  cl::sycl::free(b, ctx);
  return 0;
}
