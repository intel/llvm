// RUN: %clangxx -fsycl -O0 -fsycl-device-only -Xclang -emit-llvm -o %t.comp.ll %s
// RUN: sycl-post-link -ir-output-only -lower-esimd -S %t.comp.ll -O0 -o %t.out.ll
// RUN: FileCheck --input-file=%t.out.ll %s

// This test verifies that calls the call of slm_init<N>() that is originally
// placed inside the kernel stays in the kernel and not outlined into
// a separate spir_func functions.

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
   }).wait();
  // CHECK:     spir_kernel void @_ZTSZ4mainEUlN4sycl3_V17nd_itemILi1EEEE_()
  // CHECK-NOT: ret void
  // CHECK:     call void @llvm.genx.slm.init

  return 0;
}
