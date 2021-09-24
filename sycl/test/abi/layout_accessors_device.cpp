// RUN: %clangxx -fsycl -fsycl-device-only -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <CL/sycl.hpp>

using namespace cl::sycl;

//----------------------------------------------------------------------------//
// Global buffer accessor.
//----------------------------------------------------------------------------//

SYCL_EXTERNAL void hostAcc(accessor<int, 1, access::mode::read, access::target::global_buffer> Acc) {
  (void)Acc.get_size();
}
// CHECK:      0 | class {{.*}}::accessor<int, 1, {{.*}}::access::mode::read, {{.*}}::access::target::global_buffer, {{.*}}::access::placeholder::false_t>
// CHECK-NEXT: 0 |   class {{.*}}::detail::accessor_common<int, 1, {{.*}}::access::mode::read, {{.*}}::access::target::global_buffer, {{.*}}::access::placeholder::false_t> (base) (empty)
// CHECK-NEXT: 0 |   class {{.*}}::detail::AccessorImplDevice<1> impl
// CHECK-NEXT: 0 |     class {{.*}}::id<1> Offset
// CHECK-NEXT: 0 |       class {{.*}}::detail::array<1> (base)
// CHECK-NEXT: 0 |         size_t [1] common_array
// CHECK-NEXT: 8 |     class {{.*}}::range<1> AccessRange
// CHECK-NEXT: 8 |       class {{.*}}::detail::array<1> (base)
// CHECK-NEXT: 8 |         size_t [1] common_array
// CHECK-NEXT: 16 |     class {{.*}}::range<1> MemRange
// CHECK-NEXT: 16 |       class {{.*}}::detail::array<1> (base)
// CHECK-NEXT: 16 |         size_t [1] common_array
// CHECK-NEXT: 24 |   union {{.*}}::accessor<int, 1, {{.*}}::access::mode::read, {{.*}}::access::target::global_buffer, {{.*}}::access::placeholder::false_t>
// CHECK-NEXT: 24 |     {{.*}}::accessor<int, 1, {{.*}}::access::mode::read, {{.*}}::access::target::global_buffer, {{.*}}::access::placeholder::false_t>::ConcreteASPtrType MData
// CHECK-NEXT:     | [sizeof=32, dsize=32, align=8,
// CHECK-NEXT:     |  nvsize=32, nvalign=8]

//----------------------------------------------------------------------------//
// Local accessor.
//----------------------------------------------------------------------------//

SYCL_EXTERNAL void hostAcc(accessor<int, 1, access::mode::read_write, access::target::local> Acc) {
  (void)Acc.get_size();
}
// CHECK:      0 | class {{.*}}::accessor<int, 1, {{.*}}::access::mode::read_write, {{.*}}::access::target::local, {{.*}}::access::placeholder::false_t>
// CHECK-NEXT: 0 |   class {{.*}}::detail::accessor_common<int, 1, {{.*}}::access::mode::read_write, {{.*}}::access::target::local, {{.*}}::access::placeholder::false_t> (base) (empty)
// CHECK-NEXT: 0 |   class {{.*}}::detail::LocalAccessorBaseDevice<1> impl
// CHECK-NEXT: 0 |     class {{.*}}::range<1> AccessRange
// CHECK-NEXT: 0 |       class {{.*}}::detail::array<1> (base)
// CHECK-NEXT: 0 |         size_t [1] common_array
// CHECK-NEXT: 8 |     class {{.*}}::range<1> MemRange
// CHECK-NEXT: 8 |       class {{.*}}::detail::array<1> (base)
// CHECK-NEXT: 8 |         size_t [1] common_array
// CHECK-NEXT: 16 |     class {{.*}}::id<1> Offset
// CHECK-NEXT: 16 |       class {{.*}}::detail::array<1> (base)
// CHECK-NEXT: 16 |         size_t [1] common_array
// CHECK-NEXT: 24 |   {{.*}}::accessor<int, 1, {{.*}}::access::mode::read_write, {{.*}}::access::target::local, {{.*}}::access::placeholder::false_t>::ConcreteASPtrType MData
// CHECK-NEXT: | [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: |  nvsize=32, nvalign=8]

//----------------------------------------------------------------------------//
// Image accessor.
//----------------------------------------------------------------------------//

SYCL_EXTERNAL void hostAcc(accessor<int4, 1, access::mode::read, access::target::image> Acc) {
  (void)Acc.get_count();
}

// CHECK:      0 | class {{.*}}::accessor<class {{.*}}::vec<int, 4>, 1, {{.*}}::access::mode::read, {{.*}}::access::target::image, {{.*}}::access::placeholder::false_t>
// CHECK-NEXT: 0 |   class {{.*}}::detail::image_accessor<class {{.*}}::vec<int, 4>, 1, {{.*}}::access::mode::read, {{.*}}::access::target::image, {{.*}}::access::placeholder::false_t> (base)
// CHECK-NEXT: 0 |     {{.*}}::detail::image_accessor<class {{.*}}::vec<int, 4>, 1, {{.*}}::access::mode::read, {{.*}}::access::target::image, {{.*}}::access::placeholder::false_t>::OCLImageTy MImageObj
// CHECK-NEXT: 8 |     char [24] MPadding
// CHECK-NEXT: | [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: |  nvsize=32, nvalign=8]
