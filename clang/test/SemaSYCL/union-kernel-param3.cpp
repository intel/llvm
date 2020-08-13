// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -fcxx-exceptions -verify -fsyntax-only %s

#include "sycl.hpp"

union union_with_sampler {
  cl::sycl::sampler smpl;
  // expected-error@-1 {{'cl::sycl::sampler' cannot be used as the type of a kernel parameter}}
};

union union_with_pointer {
  int *ptr_in_union;
  // expected-error@-1 {{'int *' cannot be used as the type of a kernel parameter}}
};

template <typename KernelName, typename KernelType>
__attribute__((sycl_kernel)) void kernel_single_task(KernelType kernelFunc) {
  kernelFunc();
}

int main() {
  union_with_sampler Sampler;
  union_with_pointer Pointer;

  kernel_single_task<class kernel>([=]() {
    Sampler.smpl.use();
    int *local = Pointer.ptr_in_union;
  });
  return 0;
}
