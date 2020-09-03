// UNSUPPORTED: cuda
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -o %t.out
// TODO: enable the line below once we update NEO driver in our CI
// RUNx: %t.out

#include "../include/asmhelper.h"
#include <CL/sycl.hpp>

struct KernelFunctor {
  KernelFunctor() {}

  void operator()(cl::sycl::handler &cgh) {
    cgh.parallel_for<KernelFunctor>(
        cl::sycl::range<1>{16}, [=](cl::sycl::id<1> wiID) [[intel::reqd_sub_group_size(8)]] {
#if defined(__SYCL_DEVICE_ONLY__)
          asm volatile(".decl tmp1 v_type=G type=d num_elts=16 align=GRF\n"
                       ".decl tmp2 v_type=G type=d num_elts=16 align=GRF\n"
                       "divm (M1, 8) tmp1(0,0)<1> tmp2(0,0)<1;1,0> 0x2:f\n");
#endif
        });
  }
};

int main() {
  KernelFunctor f;
  try {
    launchInlineASMTest(f, /* sg size */ true,
                        /* exception is expected */ true);
  } catch (const cl::sycl::compile_program_error &e) {
    std::string what = e.what();
    // TODO: check for precise exception class and message once they are known
    // (pending driver update)
    if (what.find("OpenCL API failed") != std::string::npos) {
      return 0;
    }
  }
  std::cout << "Expected an exception about syntax error" << std::endl;
  return 1;
}
