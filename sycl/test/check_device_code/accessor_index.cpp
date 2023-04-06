// RUN: %clangxx -fsycl-device-only -fno-sycl-early-optimizations -S -emit-llvm -D__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ -o - %s | FileCheck %s
#include <sycl/sycl.hpp>

// Check that accessor index calculation is unrolled in headers.
// CHECK-NOT: llvm.loop
// CHECK-NOT: br i1
using namespace sycl;
int main() {
  queue Q;
  range<3> Range{8, 8, 8};
  buffer<int, 3> Buf(Range);
  Q.submit([&](handler &Cgh) {
    auto Acc = Buf.get_access<access::mode::write>(Cgh);
    local_accessor<int, 3> LocAcc(Range, Cgh);
    Cgh.parallel_for(Range, [=](item<3> It) { LocAcc[It] = Acc[It]; });
  });
}
