// RUN: not %clangxx -fsycl  %s 2>&1 | FileCheck  %s

// This test verifies call to slm_init from a function marked as
// noinline triggers an error.

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

#ifdef _MSC_VER
#define __SYCL_NOINLINE __declspec(noinline)
#else
#define __SYCL_NOINLINE __attribute__((noinline))
#endif

__SYCL_NOINLINE void bar() { slm_init(1024); }
__SYCL_NOINLINE void foo() {
  slm_init(1024);
  bar();
}

int main() {
  queue Q;
  nd_range<1> NDR{range<1>{2}, range<1>{2}};
  Q.parallel_for(NDR, [=](nd_item<1> NDI) SYCL_ESIMD_KERNEL { foo(); }).wait();
  return 0;
}
// CHECK: error: slm_init is called more than once from kernel 'typeinfo name for main::'lambda'(sycl::_V1::nd_item<1>)'.