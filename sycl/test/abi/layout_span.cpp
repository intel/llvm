// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <sycl/sycl_span.hpp>

void span(sycl::span<int, 5>) {}
// CHECK: 0 | class sycl::span<int, 5>
// CHECK-NEXT: 0 |   pointer __data
// CHECK-NEXT: | [sizeof=8, dsize=8, align=8,
// CHECK-NEXT: |  nvsize=8, nvalign=8]

//----------------------------

void span_dynamic_extent(sycl::span<int>) {}
// CHECK: 0 | class sycl::span<int>
// CHECK-NEXT: 0 |   pointer __data
// CHECK-NEXT: 8 |   size_type __size
// CHECK-NEXT: | [sizeof=16, dsize=16, align=8,
// CHECK-NEXT: |  nvsize=16, nvalign=8]
