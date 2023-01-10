// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <sycl/buffer.hpp>

void foo(sycl::buffer<int, 2>) {}

// CHECK:       0 | class sycl::buffer<int, 2, class sycl::detail::aligned_allocator<int>, void>
// CHECK:       0 | class sycl::detail::buffer_plain (base)
// CHECK-NEXT:  0 |   class std::shared_ptr<class sycl::detail::buffer_impl> impl
// CHECK-NEXT:  0 |     class std::__shared_ptr<class sycl::detail::buffer_impl, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT:  0 |       class std::__shared_ptr_access<class sycl::detail::buffer_impl, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT:  0 |       element_type * _M_ptr
// CHECK-NEXT:  8 |       class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT:  8 |         _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT:  0 | class sycl::detail::OwnerLessBase<class sycl::buffer<int, 2, class sycl::detail::aligned_allocator<int>, void> > (base) (empty)
// CHECK-NEXT: 16 |   class sycl::range<2> Range
// CHECK-NEXT: 16 |     class sycl::detail::array<2> (base)
// CHECK-NEXT: 16 |       size_t[2] common_array
// CHECK-NEXT: 32 |   size_t OffsetInBytes
// CHECK-NEXT: 40 |   _Bool IsSubBuffer
// CHECK-NEXT:    | [sizeof=48, dsize=41, align=8,
// CHECK-NEXT:    |  nvsize=41, nvalign=8]
