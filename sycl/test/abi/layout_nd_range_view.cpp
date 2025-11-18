// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// RUN: %clangxx -fsycl -fsycl-device-only -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <sycl/detail/nd_range_view.hpp>


SYCL_EXTERNAL void nd_range_view(sycl::detail::nd_range_view_v1::nd_range_view) {}
// CHECK: 0 | class sycl::detail::nd_range_view
// CHECK-NEXT: 0 |   const size_t * MGlobalSize
// CHECK-NEXT: 8 |   const size_t * MLocalSize
// CHECK-NEXT: 16 |  const size_t * MOffset
// CHECK-NEXT: 24 |  _Bool MSetNumWorkGroups
// CHECK-NEXT: 32 |  size_t MDims
// CHECK-NEXT: | [sizeof=40, dsize=40, align=8,
// CHECK-NEXT: |  nvsize=40, nvalign=8]
