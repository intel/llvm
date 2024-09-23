// RUN: %clangxx -fsycl -fsycl-device-only -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <sycl/accessor.hpp>
#include <sycl/accessor_image.hpp>

using namespace sycl;

//----------------------------------------------------------------------------//
// Global buffer accessor.
//----------------------------------------------------------------------------//

SYCL_EXTERNAL void hostAcc(accessor<int, 1, access::mode::read, access::target::global_buffer> Acc) {
  (void)Acc.get_size();
}
// CHECK:      0 | class sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::global_buffer>
// CHECK-NEXT: 0 |   class sycl::detail::accessor_common<int, 1, sycl::access::mode::read, sycl::access::target::global_buffer, sycl::access::placeholder::false_t> (base) (empty)
// CHECK-NEXT: 0 |   class sycl::detail::OwnerLessBase<class sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::global_buffer> > (base) (empty)
// CHECK-NEXT: 0 |   class sycl::detail::AccessorImplDevice<1> impl
// CHECK-NEXT: 0 |     class sycl::id<1> Offset
// CHECK-NEXT: 0 |       class sycl::detail::array<> (base)
// CHECK-NEXT: 0 |         size_t[1] common_array
// CHECK-NEXT: 8 |     class sycl::range<> AccessRange
// CHECK-NEXT: 8 |       class sycl::detail::array<> (base)
// CHECK-NEXT: 8 |         size_t[1] common_array
// CHECK-NEXT: 16 |     class sycl::range<> MemRange
// CHECK-NEXT: 16 |       class sycl::detail::array<> (base)
// CHECK-NEXT: 16 |         size_t[1] common_array
// CHECK-NEXT: 24 |   union sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::global_buffer>
// CHECK-NEXT: 24 |     ConcreteASPtrType MData
// CHECK-NEXT:     | [sizeof=32, dsize=32, align=8,
// CHECK-NEXT:     |  nvsize=32, nvalign=8]

//----------------------------------------------------------------------------//
// Local accessor.
//----------------------------------------------------------------------------//

SYCL_EXTERNAL void hostAcc(accessor<int, 1, access::mode::read_write, access::target::local> Acc) {
  (void)Acc.get_size();
}

// CHECK:      0 | class sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local>
// CHECK-NEXT: 0 |   class sycl::local_accessor_base<int, 1, sycl::access::mode::read_write, sycl::access::placeholder::false_t> (base)
// CHECK-NEXT: 0 |     class sycl::detail::accessor_common<int, 1, sycl::access::mode::read_write, sycl::access::target::local, sycl::access::placeholder::false_t> (base) (empty)
// CHECK-NEXT: 0 |     class sycl::detail::LocalAccessorBaseDevice<1> impl
// CHECK-NEXT: 0 |       class sycl::range<> AccessRange
// CHECK-NEXT: 0 |         class sycl::detail::array<> (base)
// CHECK-NEXT: 0 |           size_t[1] common_array
// CHECK-NEXT: 8 |       class sycl::range<> MemRange
// CHECK-NEXT: 8 |         class sycl::detail::array<> (base)
// CHECK-NEXT: 8 |           size_t[1] common_array
// CHECK-NEXT: 16 |       class sycl::id<1> Offset
// CHECK-NEXT: 16 |         class sycl::detail::array<> (base)
// CHECK-NEXT: 16 |           size_t[1] common_array
// CHECK-NEXT: 24 |     ConcreteASPtrType MData
// CHECK-NEXT: 0 |   class sycl::detail::OwnerLessBase<class sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> > (base) (empty)
// CHECK-NEXT: | [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: |  nvsize=32, nvalign=8]

SYCL_EXTERNAL void hostAcc(local_accessor<int, 1> Acc) {
  (void)Acc.get_size();
}

// CHECK:      0 | class sycl::local_accessor<int>
// CHECK-NEXT: 0 |   class sycl::local_accessor_base<int, 1, sycl::access::mode::read_write, sycl::access::placeholder::false_t> (base)
// CHECK-NEXT: 0 |     class sycl::detail::accessor_common<int, 1, sycl::access::mode::read_write, sycl::access::target::local, sycl::access::placeholder::false_t> (base) (empty)
// CHECK-NEXT: 0 |     class sycl::detail::LocalAccessorBaseDevice<1> impl
// CHECK-NEXT: 0 |       class sycl::range<> AccessRange
// CHECK-NEXT: 0 |         class sycl::detail::array<> (base)
// CHECK-NEXT: 0 |           size_t[1] common_array
// CHECK-NEXT: 8 |       class sycl::range<> MemRange
// CHECK-NEXT: 8 |         class sycl::detail::array<> (base)
// CHECK-NEXT: 8 |           size_t[1] common_array
// CHECK-NEXT: 16 |       class sycl::id<1> Offset
// CHECK-NEXT: 16 |         class sycl::detail::array<> (base)
// CHECK-NEXT: 16 |           size_t[1] common_array
// CHECK-NEXT: 24 |     ConcreteASPtrType MData
// CHECK-NEXT: 0 |   class sycl::detail::OwnerLessBase<class sycl::local_accessor<int> > (base) (empty)
// CHECK-NEXT: | [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: |  nvsize=32, nvalign=8]

//----------------------------------------------------------------------------//
// Image accessor.
//----------------------------------------------------------------------------//

SYCL_EXTERNAL void hostAcc(accessor<int4, 1, access::mode::read, access::target::image> Acc) {
  (void)Acc.get_count();
}

// CHECK:      0 | class sycl::accessor<class sycl::vec<int, 4>, 1, sycl::access::mode::read, sycl::access::target::image>
// CHECK-NEXT: 0 |   class sycl::detail::image_accessor<class sycl::vec<int, 4>, 1, sycl::access::mode::read, sycl::access::target::image, sycl::access::placeholder::false_t> (base)
// CHECK-NEXT: 0 |     OCLImageTy MImageObj
// CHECK-NEXT: 8 |     char[24] MPadding
// CHECK-NEXT: 0 |   class sycl::detail::OwnerLessBase<class sycl::accessor<class sycl::vec<int, 4>, 1, sycl::access::mode::read, sycl::access::target::image> > (base) (empty)
// CHECK-NEXT: | [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: |  nvsize=32, nvalign=8]
