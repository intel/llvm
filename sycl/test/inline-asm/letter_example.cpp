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
  for (int i = 0; i < problem_size; i++) {
    a[i] = i;
  }
  q.submit([&](cl::sycl::handler &cgh) {
     cgh.parallel_for<kernel_name>(
         // clang-format off
         cl::sycl::range<1>(problem_size),
     [=](cl::sycl::id<1> idx) [[intel::reqd_sub_group_size(16)]] {
    // clang-format on
#if defined(INLINE_ASM) && defined(__SYCL_DEVICE_ONLY__)
                                                 int i = idx[0];
                                                 asm volatile("{\n.decl V52 v_type=G type=d num_elts=16 align=GRF\n"
                                                              "svm_gather.4.1 (M1, 16) %0.0 V52.0\n"
                                                              "add(M1, 16) V52(0, 0)<1> V52(0, 0)<1; 1, 0> 0x1:w\n"
                                                              "svm_scatter.4.1 (M1, 16) %0.0 V52.0\n}"
                                                              :
                                                              : "rw"(&a[i]));
#else
           // clang-format off
                                                 a[idx[0]]++;
      // clang-format on
#endif
                                               });
   }).wait();

  bool currect = true;
  for (int i = 0; i < problem_size; i++) {
    if (a[i] != (i + 1)) {
      currect = false;
      std::cerr << "error in a[" << i << "]="
                << a[i] << "!=" << (i + 1) << std::endl;
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
