// UNSUPPORTED: cuda
// REQUIRES: gpu,linux
// RUN: %clangxx -fsycl %s -DINLINE_ASM -o %t.out
// RUN: %t.out
// RUN: %clangxx -fsycl %s -o %t.ref.out
// RUN: %t.ref.out



#include "include/asmhelper.h"
#include <CL/sycl.hpp>

using dataType = cl::sycl::cl_int;

template <typename T = dataType>
struct KernelFunctor : WithOutputBuffer<T> {
  KernelFunctor(size_t problem_size) : WithOutputBuffer<T>(problem_size) {}

  void operator()(cl::sycl::handler &cgh) {
    auto C = this->getOutputBuffer().template get_access<cl::sycl::access::mode::write>(cgh);
    cgh.parallel_for<KernelFunctor<T>>(
        cl::sycl::range<1>{this->getOutputBufferSize()}, [=](cl::sycl::id<1> wiID) [[cl::intel_reqd_sub_group_size(8)]] {
          int switch_field = 2;
          int switch_field1 = 2;
          int output = 0;
          switch(switch_field) {
            case 0:
              output = 9;
              break;
            case 1:
              output = 8;
              break; 
#if defined(INLINE_ASM) && defined(__SYCL_DEVICE_ONLY__)
          }
          asm volatile(".decl P1 v_type=P num_elts=8\n"
                       "cmp.ne (M1, 8) P1 %1(0,0)<0;1,0> 0x2:d\n"
                       "(P1) goto (M1, 8) label0\n"
                       "mov (M1, 8) %0(0,0)<1> 0x7:d\n"
                       "label0:"
                       : "=rw"(output) : "rw"(switch_field1));         
#else
            case 2:
              output = 7;
              break;
          }
#endif
          C[wiID] = output;
        });
  }
};

int main() {
  KernelFunctor<> f(DEFAULT_PROBLEM_SIZE);
  if (!launchInlineASMTest(f))
    return 0;

  if (verify_all_the_same(f.getOutputBufferData(), 7))
    return 0;

  return 1;
}