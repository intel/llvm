// UNSUPPORTED: cuda || hip_nvidia
// UNSUPPORTED: ze_debug-1,ze_debug4
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "../include/asmhelper.h"
#include <sycl/sycl.hpp>

struct KernelFunctor {
  KernelFunctor() {}

  void operator()(cl::sycl::handler &cgh) {
    cgh.parallel_for<KernelFunctor>(
        cl::sycl::range<1>{16}, [=
    ](cl::sycl::id<1> wiID) [[intel::reqd_sub_group_size(8)]] {
#if defined(__SYCL_DEVICE_ONLY__)
          asm volatile(".decl tmp1 v_type=G type=d num_elts=16 align=GRF\n"
                       ".decl tmp2 v_type=G type=d num_elts=16 align=GRF\n"
                       "cmp.lt (M1_NM, 8) P3 tmp1(0,0)<0;1,0> 0x3:ud\n");
#endif
        });
  }
};

int main() {
  KernelFunctor f;
  launchInlineASMTest(f, /* sg size */ true,
                      /* exception string*/ "P3: undefined predicate variable");
  return 0;
}
