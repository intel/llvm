// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// RUN: %clangxx -fsycl -fsycl-device-only -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <sycl/detail/kernel_arg_view.hpp>


SYCL_EXTERNAL void kernel_arg_view(sycl::detail::kernel_arg_view_v1::KernelArgView) {}
// CHECK: 0 | struct sycl::detail::KernelArgView
// CHECK-NEXT: 0 |   const void * MPtr
// CHECK-NEXT: 8 |   size_t MSize
// CHECK-NEXT: 16 |  kernel_param_kind_t MKind
// CHECK-NEXT: | [sizeof=24, dsize=24, align=8,
// CHECK-NEXT: |  nvsize=24, nvalign=8]
