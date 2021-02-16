// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s | FileCheck %s
// RUN: %clangxx -fsycl -fsycl-device-only -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <CL/sycl/types.hpp>

SYCL_EXTERNAL void foo(sycl::vec<int, 4>) {}

// CHECK: 0 | class sycl::vec<int, 4>
// CHECK-NEXT: 0 |   sycl::vec<int, 4>::DataType m_Data
// CHECK-NEXT: | [sizeof=16, dsize=16, align=16,
// CHECK-NEXT: |  nvsize=16, nvalign=16]

//--------------------------------------

SYCL_EXTERNAL void foo(sycl::vec<bool, 16>) {}

// CHECK: 0 | class sycl::vec<_Bool, 16>
// CHECK-NEXT: 0 |   sycl::vec<_Bool, 16>::DataType m_Data
// CHECK-NEXT: | [sizeof=16, dsize=16, align=16,
// CHECK-NEXT: |  nvsize=16, nvalign=16]
