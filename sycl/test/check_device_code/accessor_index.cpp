// RUN: %clangxx -fsycl-device-only -fno-sycl-early-optimizations -S -emit-llvm -D__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ -o - %s | FileCheck %s
#include <sycl/sycl.hpp>

// Check that accessor index calculation is unrolled in headers.
// CHECK-NOT: llvm.loop
// CHECK-NOT: br i1
using namespace sycl;

SYCL_EXTERNAL void accessor_index(accessor<int, 3, access::mode::write> Acc,
                                  local_accessor<int, 3> LocAcc, item<3> It) {
  LocAcc[It] = Acc[It];
}
