// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -fcxx-exceptions -verify -fsyntax-only %s

// This test checks if compiler reports compilation error on an attempt to pass
// accessor/sampler/pointer as SYCL kernel parameter inside union.

#include "sycl.hpp"
using namespace cl::sycl;

union union_with_sampler {
  cl::sycl::sampler smpl;
  // expected-error@-1 {{'cl::sycl::sampler' cannot be used as the type of a kernel parameter}}
};

union union_with_pointer {
  int *ptr_in_union;
  // expected-error@-1 {{'int *' cannot be used as the type of a kernel parameter}}
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {

  using Accessor =
      accessor<int, 1, access::mode::read_write, access::target::global_buffer>;

  union union_with_accessor {
    Accessor member_acc[1];
    // expected-error@-1 {{'Accessor' (aka 'accessor<int, 1, access::mode::read_write, access::target::global_buffer>') cannot be used as the type of a kernel parameter}}
  } union_acc;

  union_with_sampler Sampler;
  union_with_pointer Pointer;

  a_kernel<class kernel_A>(
      [=]() {
        Sampler.smpl.use();
      });

  a_kernel<class kernel_B>(
      [=]() {
        int *local = Pointer.ptr_in_union;
      });

  a_kernel<class kernel_C>(
      [=]() {
        union_acc.member_acc[1].use();
      });

  return 0;
}
