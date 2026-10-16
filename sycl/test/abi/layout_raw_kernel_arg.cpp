// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// RUN: %clangxx -fsycl -fsycl-device-only -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <sycl/detail/defines_elementary.hpp> // for SYCL_EXTERNAL
#include <sycl/ext/oneapi/experimental/raw_kernel_arg.hpp>

// The library reads the members of raw_kernel_arg, so its layout is pinned.
// MIsPointer is last so the other members keep their offsets.
SYCL_EXTERNAL void takeRawKernelArg(sycl::ext::oneapi::experimental::raw_kernel_arg) {}
// CHECK: 0 | class sycl::ext::oneapi::experimental::raw_kernel_arg
// CHECK-NEXT: 0 |   const void * MArgData
// CHECK-NEXT: 8 |   size_t MArgSize
// CHECK-NEXT: 16 |  _Bool MIsPointer
// CHECK-NEXT: | [sizeof=24, dsize=17, align=8,
// CHECK-NEXT: |  nvsize=17, nvalign=8]
