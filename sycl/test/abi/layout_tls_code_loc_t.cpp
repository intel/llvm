// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <sycl/detail/common.hpp>

void foo(sycl::detail::tls_code_loc_t) {}

// CHECK:      0 | class sycl::detail::tls_code_loc_t
// CHECK-NEXT: 0 |    _Bool MLocalScope
// CHECK-NEXT:   | [sizeof=1, dsize=1, align=1,
// CHECK-NEXT:   |  nvsize=1, nvalign=1]
