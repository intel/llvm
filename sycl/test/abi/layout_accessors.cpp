// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s | FileCheck %s
// REQUIRES: linux

#include <CL/sycl/accessor.hpp>

using namespace cl::sycl;

//----------------------------------------------------------------------------//
// Host buffer accessor.
//----------------------------------------------------------------------------//

void hostAcc(accessor<int, 1, access::mode::read, access::target::host_buffer> Acc) {
  (void)Acc.get_size();
}

// CHECK: 0 | class cl::sycl::accessor<int, 1, cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer, cl::sycl::access::placeholder::false_t>
// CHECK-NEXT:  0 |   class cl::sycl::detail::AccessorBaseHost (base)
// CHECK-NEXT:  0 |     class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> impl
// CHECK-NEXT:  0 |       class std::__shared_ptr<class cl::sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT:  0 |         class std::__shared_ptr_access<class cl::sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT:  0 |         std::__shared_ptr<class cl::sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT:  8 |         class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT:  8 |           _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT:  0 |   class cl::sycl::detail::accessor_common<int, 1, cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer, cl::sycl::access::placeholder::false_t> (base) (empty)
// CHECK-NEXT: 16 |   char [16] padding
// CHECK-NEXT: [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: nvsize=32, nvalign=8]

//----------------------------------------------------------------------------//
// Global buffer accessor.
//----------------------------------------------------------------------------//

void hostAcc(accessor<int, 1, access::mode::read, access::target::global_buffer> Acc) {
  (void)Acc.get_size();
}

// CHECK: 0 | class cl::sycl::accessor<int, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t>
// CHECK-NEXT: 0 |   class cl::sycl::detail::AccessorBaseHost (base)
// CHECK-NEXT: 0 |     class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> impl
// CHECK-NEXT: 0 |       class std::__shared_ptr<class cl::sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT: 0 |         class std::__shared_ptr_access<class cl::sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT: 0 |         std::__shared_ptr<class cl::sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT: 8 |         class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 8 |           _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 0 |   class cl::sycl::detail::accessor_common<int, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t> (base) (empty)
// CHECK-NEXT: 16 |   char [16] padding
// CHECK-NEXT: [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: nvsize=32, nvalign=8]

//----------------------------------------------------------------------------//
// Local accessor.
//----------------------------------------------------------------------------//

void hostAcc(accessor<int, 1, access::mode::read_write, access::target::local> Acc) {
  (void)Acc.get_size();
}

// CHECK: 0 | class cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local, cl::sycl::access::placeholder::false_t>
// CHECK-NEXT: 0 |   class cl::sycl::detail::LocalAccessorBaseHost (base)
// CHECK-NEXT: 0 |     class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost> impl
// CHECK-NEXT: 0 |       class std::__shared_ptr<class cl::sycl::detail::LocalAccessorImplHost, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT: 0 |         class std::__shared_ptr_access<class cl::sycl::detail::LocalAccessorImplHost, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT: 0 |         std::__shared_ptr<class cl::sycl::detail::LocalAccessorImplHost, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT: 8 |         class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 8 |           _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 0 |   class cl::sycl::detail::accessor_common<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local, cl::sycl::access::placeholder::false_t> (base) (empty)
// CHECK-NEXT: 16 |   char [16] padding
// CHECK-NEXT: [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: nvsize=32, nvalign=8]

//----------------------------------------------------------------------------//
// Host image accessor.
//----------------------------------------------------------------------------//

void hostAcc(accessor<int4, 1, access::mode::read_write, access::target::host_image> Acc) {
  (void)Acc.get_count();
}
// CHECK: 0 | class cl::sycl::accessor<class cl::sycl::vec<int, 4>, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::host_image, cl::sycl::access::placeholder::false_t>
// CHECK-NEXT: 0 |   class cl::sycl::detail::image_accessor<class cl::sycl::vec<int, 4>, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::host_image, cl::sycl::access::placeholder::false_t> (base)
// CHECK-NEXT: 0 |     class cl::sycl::detail::AccessorBaseHost (base)
// CHECK-NEXT: 0 |       class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> impl
// CHECK-NEXT: 0 |         class std::__shared_ptr<class cl::sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT: 0 |           class std::__shared_ptr_access<class cl::sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT: 0 |           std::__shared_ptr<class cl::sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT: 8 |           class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 8 |             _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 16 |     size_t MImageCount
// CHECK-NEXT: 24 |     enum cl::sycl::image_channel_order MImgChannelOrder
// CHECK-NEXT: 28 |     enum cl::sycl::image_channel_type MImgChannelType
// CHECK-NEXT: [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: nvsize=32, nvalign=8]

//----------------------------------------------------------------------------//
// Image accessor.
//----------------------------------------------------------------------------//

void hostAcc(accessor<int4, 1, access::mode::read, access::target::image> Acc) {
  (void)Acc.get_count();
}
// CHECK: 0 | class cl::sycl::accessor<class cl::sycl::vec<int, 4>, 1, cl::sycl::access::mode::read, cl::sycl::access::target::image, cl::sycl::access::placeholder::false_t>
// CHECK-NEXT: 0 |   class cl::sycl::detail::image_accessor<class cl::sycl::vec<int, 4>, 1, cl::sycl::access::mode::read, cl::sycl::access::target::image, cl::sycl::access::placeholder::false_t> (base)
// CHECK-NEXT: 0 |     class cl::sycl::detail::AccessorBaseHost (base)
// CHECK-NEXT: 0 |       class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> impl
// CHECK-NEXT: 0 |         class std::__shared_ptr<class cl::sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT: 0 |           class std::__shared_ptr_access<class cl::sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT: 0 |           std::__shared_ptr<class cl::sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT: 8 |           class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 8 |             _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 16 |     size_t MImageCount
// CHECK-NEXT: 24 |     enum cl::sycl::image_channel_order MImgChannelOrder
// CHECK-NEXT: 28 |     enum cl::sycl::image_channel_type MImgChannelType
// CHECK-NEXT: [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: nvsize=32, nvalign=8]
