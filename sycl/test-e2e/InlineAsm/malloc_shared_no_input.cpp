// UNSUPPORTED: cuda
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "include/asmhelper.h"
#include <iostream>
#include <sycl/sycl.hpp>

constexpr size_t problem_size = 16;

class kernel_name;

int main() {
  sycl::queue q;
  sycl::device Device = q.get_device();

  auto Vec = Device.get_info<sycl::info::device::extensions>();
  if (!isInlineASMSupported(Device) ||
      std::find(Vec.begin(), Vec.end(), "cl_intel_required_subgroup_size") ==
          std::end(Vec)) {
    std::cout << "Skipping test\n";
    return 0;
  }
  auto ctx = q.get_context();
  int *a =
      (int *)malloc_shared(sizeof(int) * problem_size, q.get_device(), ctx);
  for (int i = 0; i < problem_size; i++)
    a[i] = i;

  q.submit([&](sycl::handler &cgh) {
     cgh.parallel_for<kernel_name>(
         sycl::range<1>(problem_size),
         [=](sycl::id<1> idx) [[intel::reqd_sub_group_size(16)]] {
           int i = idx[0];
#if defined(__SYCL_DEVICE_ONLY__)
           asm volatile("mov (M1, 16) %0(0,0)<1> 0x7:d" : "=rw"(a[i]));
#else
           a[i] = 7;
#endif
         });
   }).wait();

  bool currect = true;
  for (int i = 0; i < problem_size; i++) {
    if (a[i] != 7) {
      currect = false;
      std::cerr << "error in a[" << i << "]=" << a[i] << "!=" << 7 << std::endl;
      break;
    }
  }

  if (!currect) {
    std::cerr << "Error" << std::endl;
    sycl::free(a, ctx);
    return 1;
  }

  std::cerr << "Pass" << std::endl;
  sycl::free(a, ctx);
  return 0;
}
