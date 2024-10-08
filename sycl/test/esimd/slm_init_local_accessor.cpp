// RUN: %clangxx -fsycl  %s

// This test verifies usage of slm_init and local_accessor in different kernels
// passes.

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
       scalar_load<int>(InAcc, 0);
     });
   }).wait();

  Q.submit([&](handler &CGH) {
     CGH.parallel_for(NDR, [=](nd_item<1> NDI)
                               SYCL_ESIMD_KERNEL { slm_init(1024); });
   }).wait();

  return 0;
}
