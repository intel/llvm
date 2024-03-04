// UNSUPPORTED: cuda, hip
// REQUIRES: gpu,linux,sg-16,aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "include/asmhelper.h"
#include <iostream>
#include <sycl/sycl.hpp>

constexpr size_t problem_size = 100;

class kernel_name;

int main() {
  sycl::queue q;

  sycl::device Device = q.get_device();

  if (!isInlineASMSupported(Device)) {
    std::cout << "Skipping test\n";
    return 0;
  }

  auto ctx = q.get_context();
  int *a =
      (int *)malloc_shared(sizeof(int) * problem_size, q.get_device(), ctx);
  int *b =
      (int *)malloc_shared(sizeof(int) * problem_size, q.get_device(), ctx);
  for (int i = 0; i < problem_size; i++) {
    b[i] = -1;
    a[i] = i;
  }

  q.submit([&](sycl::handler &cgh) {
     cgh.parallel_for<kernel_name>(
         sycl::range<1>(problem_size),
         [=](sycl::id<1> idx) [[sycl::reqd_sub_group_size(16)]] {
           int i = idx[0];
           volatile int tmp = a[i];
           tmp += 1;
#if defined(__SYCL_DEVICE_ONLY__)
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
      std::cerr << "error in a[" << i << "]=" << b[i] << "!=" << a[i]
                << std::endl;
      break;
    }
  }

  if (!currect) {
    std::cerr << "Error" << std::endl;
    sycl::free(a, ctx);
    sycl::free(b, ctx);
    return 1;
  }

  std::cerr << "Pass" << std::endl;
  sycl::free(a, ctx);
  sycl::free(b, ctx);
  return 0;
}
