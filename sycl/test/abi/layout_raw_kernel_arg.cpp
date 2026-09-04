// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// RUN: %clangxx -fsycl -fsycl-device-only -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <sycl/detail/defines_elementary.hpp> // for SYCL_EXTERNAL
#include <sycl/ext/oneapi/experimental/raw_kernel_arg.hpp>

// A raw_kernel_arg crosses the ABI boundary as an element of the span that the
// nd_launch overloads take, and the graph extension copies one as bytes, so its
// layout is fixed here. MIsPointer comes last, so that the two members the
// library read before it existed keep their offsets.
SYCL_EXTERNAL void takeRawKernelArg(sycl::ext::oneapi::experimental::raw_kernel_arg) {}
// CHECK: 0 | class sycl::ext::oneapi::experimental::raw_kernel_arg
// CHECK-NEXT: 0 |   const void * MArgData
// CHECK-NEXT: 8 |   size_t MArgSize
// CHECK-NEXT: 16 |  _Bool MIsPointer
// CHECK-NEXT: | [sizeof=24, dsize=17, align=8,
// CHECK-NEXT: |  nvsize=17, nvalign=8]
