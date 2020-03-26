// UNSUPPORTED: cuda
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -DINLINE_ASM -o %t.out
// RUN: %t.out

#include "include/asmcheck.h"
#include <CL/sycl.hpp>
#include <iostream>
#define N 100
using namespace cl::sycl;
int main() {
  int *a;
  queue q;
  sycl::device Device = q.get_device();

  if (!isInlineASMSupported(Device) || !Device.has_extension("cl_intel_required_subgroup_size")) {
    std::cout << "Skipping test\n";
    return 0;
  }
  auto ctx = q.get_context();
  a = (int *)malloc_shared(sizeof(int) * N, q.get_device(), ctx);
  for (int i = 0; i < N; i++)
    a[i] = i;
  q.submit([&](handler &cgh) {
     cgh.parallel_for<class kernel>(
         range<1>(N), [=](id<1> idx)
                          [[cl::intel_reqd_sub_group_size(16)]] {
#if defined(INLINE_ASM) && defined(__SYCL_DEVICE_ONLY__)
                            int i = idx[0];
                            asm volatile("{\n.decl V52 v_type=G type=d num_elts=16 align=GRF\n"
                                         "svm_gather.4.1 (M1, 16) %0.0 V52.0\n"
                                         "add(M1, 16) V52(0, 0)<1> V52(0, 0)<1; 1, 0> 0x1:w\n"
                                         "svm_scatter.4.1 (M1, 16) %0.0 V52.0\n}"
                                         :
                                         : "rw"(&a[i]));
#else
                            a[idx[0]]++;
#endif
                          });
   }).wait();

  bool currect = true;
  for (int i = 0; i < N; i++) {
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
    return -1;
  }
  std::cerr << "Pass" << std::endl;
  cl::sycl::free(a, ctx);
  return 0;
}
