//RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only %s

// This test checks if compiler reports compilation error on an attempt to pass
// accessor/sampler as SYCL kernel parameter inside union.

#include "Inputs/sycl.hpp"
using namespace cl::sycl;

union union_with_sampler {
  cl::sycl::sampler smpl;
  // expected-error@-1 {{'cl::sycl::sampler' cannot be used inside a union kernel parameter}}
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {

  using Accessor =
      accessor<int, 1, access::mode::read_write, access::target::global_buffer>;

  union union_with_accessor {
    Accessor member_acc[1];
    // expected-error@-1 {{'Accessor' (aka 'accessor<int, 1, access::mode::read_write, access::target::global_buffer>') cannot be used inside a union kernel parameter}}
  } union_acc;

  union_with_sampler Sampler;

  a_kernel<class kernel_A>(
      [=]() {
        Sampler.smpl.use();
      });

  a_kernel<class kernel_B>(
      [=]() {
        union_acc.member_acc[1].use();
      });

  return 0;
}
