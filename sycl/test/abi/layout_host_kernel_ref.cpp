// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <sycl/detail/cg_types.hpp>

void foo(sycl::detail::HostKernelRefBase *) {}

// CHECK:      0 | class sycl::detail::HostKernelRefBase
// CHECK-NEXT: 0 |   class sycl::detail::HostKernelBase (primary base)
// CHECK-NEXT: 0 |     (HostKernelBase vtable pointer)
// CHECK-NEXT:   | [sizeof=8, dsize=8, align=8,
// CHECK-NEXT:   |  nvsize=8, nvalign=8]
