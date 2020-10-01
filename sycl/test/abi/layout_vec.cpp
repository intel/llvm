// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s | FileCheck %s
// REQUIRES: linux

// clang-format off

#include <CL/sycl/types.hpp>

void foo(sycl::vec<int, 4>) {}

// CHECK: 0 | class cl::sycl::vec<int, 4>
// CHECK-NEXT: 0 |   cl::sycl::vec<int, 4>::DataType m_Data
// CHECK-NEXT: | [sizeof=16, dsize=16, align=16,
// CHECK-NEXT: |  nvsize=16, nvalign=16]

//--------------------------------------

void foo(sycl::vec<bool, 16>) {}

// CHECK: 0 | class cl::sycl::vec<_Bool, 16>
// CHECK-NEXT: 0 |   cl::sycl::vec<_Bool, 16>::DataType m_Data
// CHECK-NEXT: | [sizeof=16, dsize=16, align=16,
// CHECK-NEXT: |  nvsize=16, nvalign=16]
