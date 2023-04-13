// UNSUPPORTED: cuda || hip_nvidia
// UNSUPPORTED: ze_debug-1,ze_debug4
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "../include/asmhelper.h"
#include <sycl/sycl.hpp>

struct KernelFunctor {
  KernelFunctor() {}

  void operator()(sycl::handler &cgh) {
    cgh.parallel_for<KernelFunctor>(
        sycl::range<1>{16},
        [=](sycl::id<1> wiID) [[intel::reqd_sub_group_size(16)]] {
#if defined(__SYCL_DEVICE_ONLY__)
          asm volatile(".decl tmp1 v_type=G type=d num_elts=16 align=GRF\n"
                       ".decl tmp2 v_type=G type=d num_elts=16 align=GRF\n"
                       "goto (M1, 16) check_label0\n");
#endif
        });
  }
};

int main() {
  KernelFunctor f;
  launchInlineASMTest(f, /* sg size */ true, /* exception expected */ true);
  return 0;
}
