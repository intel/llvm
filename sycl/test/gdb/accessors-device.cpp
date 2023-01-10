// RUN: %clangxx -fsycl-device-only -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -I %sycl_include -Wno-sycl-strict | FileCheck %s
// UNSUPPORTED: windows
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue q;
  buffer<int, 1> b(1);
  q.submit([&](handler &cgh) {
    accessor a{b, cgh};

    cgh.single_task([=]() { a[0] = 42; });
  });
}

// AccessorImplDevice must have MemRange and Offset fields

// CHECK:           0 | class sycl::detail::AccessorImplDevice<1>
// CHECK-NEXT:      0 |   class sycl::id<1> Offset
// CHECK-NEXT:      0 |     class sycl::detail::array<1> (base)
// CHECK-NEXT:      0 |       size_t[1] common_array
// CHECK-NEXT:      8 |   class sycl::range<1> AccessRange
// CHECK-NEXT:      8 |     class sycl::detail::array<1> (base)
// CHECK-NEXT:      8 |       size_t[1] common_array
// CHECK-NEXT:     16 |   class sycl::range<1> MemRange
// CHECK-NEXT:     16 |     class sycl::detail::array<1> (base)
// CHECK-NEXT:     16 |       size_t[1] common_array
// CHECK-NEXT:        | [sizeof=24, dsize=24, align=8,
// CHECK-NEXT:        |  nvsize=24, nvalign=8]

// accessor.impl must be present and of AccessorImplDevice type

// CHECK:           0 | class sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer, sycl::access::placeholder::false_t>
// CHECK-NEXT:      0 |   class sycl::detail::accessor_common<int, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer, sycl::access::placeholder::false_t> (base) (empty)
// CHECK-NEXT:      0 |   class sycl::detail::OwnerLessBase<class sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer, sycl::access::placeholder::false_t> > (base) (empty)
// CHECK-NEXT:      0 |   class sycl::detail::AccessorImplDevice<1> impl
// CHECK-NEXT:      0 |     class sycl::id<1> Offset
// CHECK-NEXT:      0 |       class sycl::detail::array<1> (base)
// CHECK-NEXT:      0 |         size_t[1] common_array
// CHECK-NEXT:      8 |     class sycl::range<1> AccessRange
// CHECK-NEXT:      8 |       class sycl::detail::array<1> (base)
// CHECK-NEXT:      8 |         size_t[1] common_array
// CHECK-NEXT:     16 |     class sycl::range<1> MemRange
// CHECK-NEXT:     16 |       class sycl::detail::array<1> (base)
// CHECK-NEXT:     16 |         size_t[1] common_array
// CHECK-NEXT:     24 |   union sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer, sycl::access::placeholder::false_t>::(anonymous at
// CHECK-NEXT:     24 |     ConcreteASPtrType MData
// CHECK-NEXT:        | [sizeof=32, dsize=32, align=8,
// CHECK-NEXT:        |  nvsize=32, nvalign=8]
