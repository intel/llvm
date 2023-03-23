// RUN: %clangxx -fsycl-device-only -fno-sycl-early-optimizations -S -emit-llvm -o - %s | FileCheck %s
#include <sycl/sycl.hpp>

// Check that accessor index calculation is unrolled in headers.
// CHECK-NOT: llvm.loop
using namespace sycl;
int main() {
  queue Q;
  range<3> Range{8, 8, 8};
  buffer<int, 3> Buf(Range);
  Q.submit([&](handler &Cgh) {
    auto Acc = Buf.get_access<access::mode::write>(Cgh);
    local_accessor<int, 3> locAcc(Range, Cgh);
    Cgh.parallel_for(Range, [=](item<3> it) { locAcc[it] = Acc[it]; });
  });
}
