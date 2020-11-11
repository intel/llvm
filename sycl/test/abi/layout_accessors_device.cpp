// RUN: %clangxx -fsycl -fsycl-device-only -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s | FileCheck %s
// REQUIRES: linux

// clang-format off

#include <CL/sycl/accessor.hpp>

using namespace cl::sycl;

//----------------------------------------------------------------------------//
// Global buffer accessor.
//----------------------------------------------------------------------------//

SYCL_EXTERNAL void hostAcc(accessor<int, 1, access::mode::read, access::target::global_buffer> Acc) {
  (void)Acc.get_size();
}
// CHECK:  0 | class cl::sycl::accessor<int, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t, class cl::sycl::ONEAPI::accessor_property_list<> >
// CHECK-NEXT: 0 |   class cl::sycl::detail::accessor_common<int, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t, class cl::sycl::ONEAPI::accessor_property_list<> > (base) (empty)
// CHECK-NEXT: 0 |   class cl::sycl::detail::AccessorImplDevice<1> impl
// CHECK-NEXT: 0 |     class cl::sycl::id<1> Offset
// CHECK-NEXT: 0 |       class cl::sycl::detail::array<1> (base)
// CHECK-NEXT: 0 |         size_t [1] common_array
// CHECK-NEXT: 8 |     class cl::sycl::range<1> AccessRange
// CHECK-NEXT: 8 |       class cl::sycl::detail::array<1> (base)
// CHECK-NEXT: 8 |         size_t [1] common_array
// CHECK-NEXT: 16 |     class cl::sycl::range<1> MemRange
// CHECK-NEXT: 16 |       class cl::sycl::detail::array<1> (base)
// CHECK-NEXT: 16 |         size_t [1] common_array
// CHECK-NEXT: 24 |   union cl::sycl::accessor<int, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t, class cl::sycl::ONEAPI::accessor_property_list<> > 
// CHECK-NEXT: 24 |     cl::sycl::accessor<int, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t, class cl::sycl::ONEAPI::accessor_property_list<> >::ConcreteASPtrType MData
// CHECK-NEXT:     | [sizeof=32, dsize=32, align=8,
// CHECK-NEXT:     |  nvsize=32, nvalign=8]

//----------------------------------------------------------------------------//
// Local accessor.
//----------------------------------------------------------------------------//

SYCL_EXTERNAL void hostAcc(accessor<int, 1, access::mode::read_write, access::target::local> Acc) {
  (void)Acc.get_size();
}
// CHECK: 0 | class cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local, cl::sycl::access::placeholder::false_t, class cl::sycl::ONEAPI::accessor_property_list<> >
// CHECK-NEXT: 0 |   class cl::sycl::detail::accessor_common<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local, cl::sycl::access::placeholder::false_t, class cl::sycl::ONEAPI::accessor_property_list<> > (base) (empty)
// CHECK-NEXT: 0 |   class cl::sycl::detail::LocalAccessorBaseDevice<1> impl
// CHECK-NEXT: 0 |     class cl::sycl::range<1> AccessRange
// CHECK-NEXT: 0 |       class cl::sycl::detail::array<1> (base)
// CHECK-NEXT: 0 |         size_t [1] common_array
// CHECK-NEXT: 8 |     class cl::sycl::range<1> MemRange
// CHECK-NEXT: 8 |       class cl::sycl::detail::array<1> (base)
// CHECK-NEXT: 8 |         size_t [1] common_array
// CHECK-NEXT: 16 |     class cl::sycl::id<1> Offset
// CHECK-NEXT: 16 |       class cl::sycl::detail::array<1> (base)
// CHECK-NEXT: 16 |         size_t [1] common_array
// CHECK-NEXT: 24 |   cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local, cl::sycl::access::placeholder::false_t, class cl::sycl::ONEAPI::accessor_property_list<> >::ConcreteASPtrType MData
// CHECK-NEXT: | [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: |  nvsize=32, nvalign=8]

//----------------------------------------------------------------------------//
// Image accessor.
//----------------------------------------------------------------------------//

SYCL_EXTERNAL void hostAcc(accessor<int4, 1, access::mode::read, access::target::image> Acc) {
  (void)Acc.get_count();
}

// CHECK: 0 | class cl::sycl::accessor<class cl::sycl::vec<int, 4>, 1, cl::sycl::access::mode::read, cl::sycl::access::target::image, cl::sycl::access::placeholder::false_t, class cl::sycl::ONEAPI::accessor_property_list<> >
// CHECK-NEXT: 0 |   class cl::sycl::detail::image_accessor<class cl::sycl::vec<int, 4>, 1, cl::sycl::access::mode::read, cl::sycl::access::target::image, cl::sycl::access::placeholder::false_t> (base)
// CHECK-NEXT: 0 |     cl::sycl::detail::image_accessor<class cl::sycl::vec<int, 4>, 1, cl::sycl::access::mode::read, cl::sycl::access::target::image, cl::sycl::access::placeholder::false_t>::OCLImageTy MImageObj
// CHECK-NEXT: 8 |     char [24] MPadding
// CHECK-NEXT: | [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: |  nvsize=32, nvalign=8]
