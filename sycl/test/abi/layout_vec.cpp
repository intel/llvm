// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// RUN: %clangxx -fsycl -fsycl-device-only -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s

// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <sycl/types.hpp>

SYCL_EXTERNAL void foo(sycl::vec<int, 4>) {}

// CHECK: 0 | class sycl::vec<int, 4>
// CHECK: 0 |   class sycl::detail::vec_arith<int, 4> (base) (empty)
// CHECK-NEXT: 0 |     class sycl::detail::vec_arith_common<int, 4> (base) (empty)
// CHECK-NEXT: 0 |   struct std::array<int, 4> m_Data
// CHECK-NEXT: 0 |     typename _AT_Type::_Type _M_elems
// CHECK-NEXT: | [sizeof=16, dsize=16, align=16,
// CHECK-NEXT: |  nvsize=16, nvalign=16]

//--------------------------------------

SYCL_EXTERNAL void foo(sycl::vec<bool, 16>) {}

// CHECK: 0 | class sycl::vec<_Bool, 16>
// CHECK: 0 |   class sycl::detail::vec_arith<_Bool, 16> (base) (empty)
// CHECK-NEXT: 0 |     class sycl::detail::vec_arith_common<_Bool, 16> (base) (empty)
// CHECK-NEXT: 0 |   struct std::array<_Bool, 16> m_Data
// CHECK-NEXT: 0 |     typename _AT_Type::_Type _M_elems
// CHECK-NEXT: | [sizeof=16, dsize=16, align=16,
// CHECK-NEXT: |  nvsize=16, nvalign=16]
