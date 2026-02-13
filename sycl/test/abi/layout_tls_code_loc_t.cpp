// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <sycl/detail/common.hpp>

void foo(sycl::detail::tls_code_loc_t) {}

// CHECK:      0 | class sycl::detail::tls_code_loc_t
// CHECK-NEXT: 0 |   detail::code_location & CodeLocTLSRef
// CHECK-NEXT: 8 |   _Bool MLocalScope
// CHECK-NEXT:   | [sizeof=16, dsize=9, align=8,
// CHECK-NEXT:   |  nvsize=9, nvalign=8]
