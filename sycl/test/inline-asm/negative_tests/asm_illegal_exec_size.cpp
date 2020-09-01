// UNSUPPORTED: cuda
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -DINLINE_ASM -o %t.out
// TODO: enable the line below once we update NEO driver in our CI
// RUNx: %t.out

#include "include/asmhelper.h"
#include <CL/sycl.hpp>

using dataType = cl::sycl::cl_int;

template <typename T = dataType>
struct KernelFunctor : WithOutputBuffer<T> {
  KernelFunctor(size_t problem_size) : WithOutputBuffer<T>(problem_size) {}

  void operator()(cl::sycl::handler &cgh) {
    cgh.parallel_for<KernelFunctor<T>>(
        cl::sycl::range<1>{this->getOutputBufferSize()}, [=](cl::sycl::id<1> wiID) [[intel::reqd_sub_group_size(8)]] {
#if defined(INLINE_ASM) && defined(__SYCL_DEVICE_ONLY__)
          asm volatile(".decl tmp1 v_type=G type=d num_elts=16 align=GRF\n"
                       ".decl tmp2 v_type=G type=d num_elts=16 align=GRF\n"
                       "mov (M1_NM, 6) tmp1(0,1)<1>  tmp2(0,0)<1;1,0>\n");
#endif
        });
  }
};

int main() {
  KernelFunctor<> f(DEFAULT_PROBLEM_SIZE);
  try {
    launchInlineASMTest(f, /* sg size */ true,
                        /* exception is expected */ true);
  } catch (const cl::sycl::compile_program_error &e) {
    std::string what = e.what();
    // TODO: check for precise exception class and message once they are known
    // (pending driver update)
    if (what.find("invalid execution size") == std::string::npos) {
      std::cout << "Expected an exception about syntax error" << std::endl;
      return 1;
    }
  }
  return 0;
}
