// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s | FileCheck %s
// REQUIRES: linux

// clang-format off

// TODO fix individual headers and include them instead of sycl.hpp
#include <CL/sycl.hpp>


void range(sycl::range<2>) {}

// CHECK: 0 | class cl::sycl::range<2>
// CHECK-NEXT: 0 |   class cl::sycl::detail::array<2> (base)
// CHECK-NEXT: 0 |     size_t [2] common_array
// CHECK-NEXT: | [sizeof=16, dsize=16, align=8,
// CHECK-NEXT: |  nvsize=16, nvalign=8]

//----------------------------

void id(sycl::id<2>) {}

// CHECK: 0 | class cl::sycl::id<3>
// CHECK-NEXT: 0 |   class cl::sycl::detail::array<3> (base)
// CHECK-NEXT: 0 |     size_t [3] common_array
// CHECK-NEXT: | [sizeof=24, dsize=24, align=8,
// CHECK-NEXT: |  nvsize=24, nvalign=8]

//----------------------------

void item(sycl::item<2>) {}

// CHECK: 0 | class cl::sycl::item<2, true>
// CHECK-NEXT: 0 |   struct cl::sycl::detail::ItemBase<2, true> MImpl
// CHECK-NEXT: 0 |     class cl::sycl::range<2> MExtent
// CHECK-NEXT: 0 |       class cl::sycl::detail::array<2> (base)
// CHECK-NEXT: 0 |         size_t [2] common_array
// CHECK-NEXT: 16 |     class cl::sycl::id<2> MIndex
// CHECK-NEXT: 16 |       class cl::sycl::detail::array<2> (base)
// CHECK-NEXT: 16 |         size_t [2] common_array
// CHECK-NEXT: 32 |     class cl::sycl::id<2> MOffset
// CHECK-NEXT: 32 |       class cl::sycl::detail::array<2> (base)
// CHECK-NEXT: 32 |         size_t [2] common_array
// CHECK-NEXT: | [sizeof=48, dsize=48, align=8,
// CHECK-NEXT: |  nvsize=48, nvalign=8]

//----------------------------

void nd_item(sycl::nd_item<2>) {}

// CHECK: 0 | class cl::sycl::nd_item<2>
// CHECK-NEXT: 0 |   class cl::sycl::item<2, true> globalItem
// CHECK-NEXT: 0 |     struct cl::sycl::detail::ItemBase<2, true> MImpl
// CHECK-NEXT: 0 |       class cl::sycl::range<2> MExtent
// CHECK-NEXT: 0 |         class cl::sycl::detail::array<2> (base)
// CHECK-NEXT: 0 |           size_t [2] common_array
// CHECK-NEXT: 16 |       class cl::sycl::id<2> MIndex
// CHECK-NEXT: 16 |         class cl::sycl::detail::array<2> (base)
// CHECK-NEXT: 16 |           size_t [2] common_array
// CHECK-NEXT: 32 |       class cl::sycl::id<2> MOffset
// CHECK-NEXT: 32 |         class cl::sycl::detail::array<2> (base)
// CHECK-NEXT: 32 |           size_t [2] common_array
// CHECK-NEXT: 48 |   class cl::sycl::item<2, false> localItem
// CHECK-NEXT: 48 |     struct cl::sycl::detail::ItemBase<2, false> MImpl
// CHECK-NEXT: 48 |       class cl::sycl::range<2> MExtent
// CHECK-NEXT: 48 |         class cl::sycl::detail::array<2> (base)
// CHECK-NEXT: 48 |           size_t [2] common_array
// CHECK-NEXT: 64 |       class cl::sycl::id<2> MIndex
// CHECK-NEXT: 64 |         class cl::sycl::detail::array<2> (base)
// CHECK-NEXT: 64 |           size_t [2] common_array
// CHECK-NEXT: 80 |   class cl::sycl::group<2> Group
// CHECK-NEXT: 80 |     class cl::sycl::range<2> globalRange
// CHECK-NEXT: 80 |       class cl::sycl::detail::array<2> (base)
// CHECK-NEXT: 80 |         size_t [2] common_array
// CHECK-NEXT: 96 |     class cl::sycl::range<2> localRange
// CHECK-NEXT: 96 |       class cl::sycl::detail::array<2> (base)
// CHECK-NEXT: 96 |         size_t [2] common_array
// CHECK-NEXT: 112 |     class cl::sycl::range<2> groupRange
// CHECK-NEXT: 112 |       class cl::sycl::detail::array<2> (base)
// CHECK-NEXT: 112 |         size_t [2] common_array
// CHECK-NEXT: 128 |     class cl::sycl::id<2> index
// CHECK-NEXT: 128 |       class cl::sycl::detail::array<2> (base)
// CHECK-NEXT: 128 |         size_t [2] common_array
// CHECK-NEXT:     | [sizeof=144, dsize=144, align=8,
// CHECK-NEXT:     |  nvsize=144, nvalign=8]
