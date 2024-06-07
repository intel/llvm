// UNSUPPORTED: cuda || hip
// UNSUPPORTED: ze_debug
// REQUIRES: gpu,linux,sg-16
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "../include/asmhelper.h"
#include <sycl/detail/core.hpp>

struct KernelFunctor {
  KernelFunctor() {}

  void operator()(sycl::handler &cgh) {
    cgh.parallel_for<KernelFunctor>(
        sycl::range<1>{16},
        [=](sycl::id<1> wiID) [[sycl::reqd_sub_group_size(16)]] {
#if defined(__SYCL_DEVICE_ONLY__)
          asm volatile(".decl tmp1 v_type=G type=d num_elts=16 align=GRF\n"
                       ".decl tmp2 v_type=G type=d num_elts=16 align=GRF\n"
                       "mov (M1_NM, 16) tmp1(0,1)<1>  tmp2(0,0)\n");
#endif
        });
  }
};

int main() {
  KernelFunctor f;
  launchInlineASMTest(f, {16}, /* exception expected */ true);
  return 0;
}
