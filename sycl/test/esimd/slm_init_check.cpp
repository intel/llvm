// RUN: not %clangxx -fsycl  %s 2>&1 | FileCheck  %s

// This test verifies more than 1 call to slm_init triggers an error.

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

int main() {
  queue Q;
  nd_range<1> NDR{range<1>{2}, range<1>{2}};
  Q.parallel_for(NDR, [=](nd_item<1> NDI) SYCL_ESIMD_KERNEL {
     slm_init(1024);
     slm_init(1024);
   }).wait();
  // CHECK: error: slm_init is called more than once from kernel 'typeinfo name for main::'lambda'(sycl::_V1::nd_item<1>)'.

  return 0;
}
