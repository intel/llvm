// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s | FileCheck %s
// RUN: %clangxx -fsycl -fsycl-device-only -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

// TODO fix individual headers and include them instead of sycl.hpp
#include <CL/sycl.hpp>


SYCL_EXTERNAL void id(sycl::id<2>) {}

// CHECK: 0 | class sycl::id<2>
// CHECK-NEXT: 0 |   class sycl::detail::array<2> (base)
// CHECK-NEXT: 0 |     size_t [2] common_array
// CHECK-NEXT: | [sizeof=16, dsize=16, align=8,
// CHECK-NEXT: |  nvsize=16, nvalign=8]

//----------------------------

SYCL_EXTERNAL void range(sycl::range<2>) {}

// CHECK: 0 | class sycl::range<2>
// CHECK-NEXT: 0 |   class sycl::detail::array<2> (base)
// CHECK-NEXT: 0 |     size_t [2] common_array
// CHECK-NEXT: | [sizeof=16, dsize=16, align=8,
// CHECK-NEXT: |  nvsize=16, nvalign=8]

//----------------------------

SYCL_EXTERNAL void item(sycl::item<2>) {}

// CHECK: 0 | class sycl::item<2, true>
// CHECK-NEXT: 0 |   struct sycl::detail::ItemBase<2, true> MImpl
// CHECK-NEXT: 0 |     class sycl::range<2> MExtent
// CHECK-NEXT: 0 |       class sycl::detail::array<2> (base)
// CHECK-NEXT: 0 |         size_t [2] common_array
// CHECK-NEXT: 16 |     class sycl::id<2> MIndex
// CHECK-NEXT: 16 |       class sycl::detail::array<2> (base)
// CHECK-NEXT: 16 |         size_t [2] common_array
// CHECK-NEXT: 32 |     class sycl::id<2> MOffset
// CHECK-NEXT: 32 |       class sycl::detail::array<2> (base)
// CHECK-NEXT: 32 |         size_t [2] common_array
// CHECK-NEXT: | [sizeof=48, dsize=48, align=8,
// CHECK-NEXT: |  nvsize=48, nvalign=8]

//----------------------------

SYCL_EXTERNAL void nd_item(sycl::nd_item<2>) {}

// CHECK: 0 | class sycl::nd_item<2>
// CHECK-NEXT: 0 |   class sycl::item<2, true> globalItem
// CHECK-NEXT: 0 |     struct sycl::detail::ItemBase<2, true> MImpl
// CHECK-NEXT: 0 |       class sycl::range<2> MExtent
// CHECK-NEXT: 0 |         class sycl::detail::array<2> (base)
// CHECK-NEXT: 0 |           size_t [2] common_array
// CHECK-NEXT: 16 |       class sycl::id<2> MIndex
// CHECK-NEXT: 16 |         class sycl::detail::array<2> (base)
// CHECK-NEXT: 16 |           size_t [2] common_array
// CHECK-NEXT: 32 |       class sycl::id<2> MOffset
// CHECK-NEXT: 32 |         class sycl::detail::array<2> (base)
// CHECK-NEXT: 32 |           size_t [2] common_array
// CHECK-NEXT: 48 |   class sycl::item<2, false> localItem
// CHECK-NEXT: 48 |     struct sycl::detail::ItemBase<2, false> MImpl
// CHECK-NEXT: 48 |       class sycl::range<2> MExtent
// CHECK-NEXT: 48 |         class sycl::detail::array<2> (base)
// CHECK-NEXT: 48 |           size_t [2] common_array
// CHECK-NEXT: 64 |       class sycl::id<2> MIndex
// CHECK-NEXT: 64 |         class sycl::detail::array<2> (base)
// CHECK-NEXT: 64 |           size_t [2] common_array
// CHECK-NEXT: 80 |   class sycl::group<2> Group
// CHECK-NEXT: 80 |     class sycl::range<2> globalRange
// CHECK-NEXT: 80 |       class sycl::detail::array<2> (base)
// CHECK-NEXT: 80 |         size_t [2] common_array
// CHECK-NEXT: 96 |     class sycl::range<2> localRange
// CHECK-NEXT: 96 |       class sycl::detail::array<2> (base)
// CHECK-NEXT: 96 |         size_t [2] common_array
// CHECK-NEXT: 112 |     class sycl::range<2> groupRange
// CHECK-NEXT: 112 |       class sycl::detail::array<2> (base)
// CHECK-NEXT: 112 |         size_t [2] common_array
// CHECK-NEXT: 128 |     class sycl::id<2> index
// CHECK-NEXT: 128 |       class sycl::detail::array<2> (base)
// CHECK-NEXT: 128 |         size_t [2] common_array
// CHECK-NEXT:     | [sizeof=144, dsize=144, align=8,
// CHECK-NEXT:     |  nvsize=144, nvalign=8]

//----------------------------

SYCL_EXTERNAL void nd_range(sycl::nd_range<2>) {}
// CHECK: 0 | class sycl::nd_range<2>
// CHECK-NEXT: 0 |   class sycl::range<2> globalSize
// CHECK-NEXT: 0 |     class sycl::detail::array<2> (base)
// CHECK-NEXT: 0 |       size_t [2] common_array
// CHECK-NEXT: 16 |   class sycl::range<2> localSize
// CHECK-NEXT: 16 |     class sycl::detail::array<2> (base)
// CHECK-NEXT: 16 |       size_t [2] common_array
// CHECK-NEXT: 32 |   class sycl::id<2> offset
// CHECK-NEXT: 32 |     class sycl::detail::array<2> (base)
// CHECK-NEXT: 32 |       size_t [2] common_array
// CHECK-NEXT: | [sizeof=48, dsize=48, align=8,
// CHECK-NEXT: |  nvsize=48, nvalign=8]
