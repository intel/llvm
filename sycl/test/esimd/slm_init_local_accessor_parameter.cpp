// RUN: not %clangxx -fsycl  %s 2>&1 | FileCheck  %s

// This test verifies usage of slm_init and local_accessor triggers an error.

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

int main() {
  queue Q;
  nd_range<1> NDR{range<1>{2}, range<1>{2}};
  Q.submit([&](handler &CGH) {
     auto InAcc = local_accessor<int, 1>(5, CGH);
     CGH.parallel_for(NDR, [=](nd_item<1> NDI) SYCL_ESIMD_KERNEL {
       slm_init(1024);
       scalar_load<int>(InAcc, 0);
     });
   }).wait();
  // CHECK: error: slm_init can not be used with local accessors.

  return 0;
}
