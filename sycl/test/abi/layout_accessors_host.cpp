// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <sycl/accessor.hpp>

using namespace sycl;

//----------------------------------------------------------------------------//
// Host buffer accessor.
//----------------------------------------------------------------------------//

void hostAcc(accessor<int, 1, access::mode::read, access::target::host_buffer> Acc) {
  (void)Acc.get_size();
}

// CHECK-COUNT-2: 0 | class sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::host_buffer, sycl::access::placeholder::false_t>
// CHECK-NEXT:  0 |   class sycl::detail::AccessorBaseHost (base)
// CHECK-NEXT:  0 |     class std::shared_ptr<class sycl::detail::AccessorImplHost> impl
// CHECK-NEXT:  0 |       class std::__shared_ptr<class sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT:  0 |         class std::__shared_ptr_access<class sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT:  0 |         element_type * _M_ptr
// CHECK-NEXT:  8 |         class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT:  8 |           _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT:  0 |   class sycl::detail::accessor_common<int, 1, sycl::access::mode::read, sycl::access::target::host_buffer, sycl::access::placeholder::false_t> (base) (empty)
// CHECK-NEXT:  0 |   class sycl::detail::OwnerLessBase<class sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::host_buffer, sycl::access::placeholder::false_t> > (base) (empty)
// CHECK-NEXT: 16 | detail::AccHostDataT * MAccData
// CHECK-NEXT: 24 |   char[8] padding
// CHECK-NEXT: [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: nvsize=32, nvalign=8]

//----------------------------------------------------------------------------//
// Global buffer accessor.
//----------------------------------------------------------------------------//

void hostAcc(accessor<int, 1, access::mode::read, access::target::global_buffer> Acc) {
  (void)Acc.get_size();
}

// CHECK:      0 | class sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::global_buffer, sycl::access::placeholder::false_t>
// CHECK-NEXT: 0 |   class sycl::detail::AccessorBaseHost (base)
// CHECK-NEXT: 0 |     class std::shared_ptr<class sycl::detail::AccessorImplHost> impl
// CHECK-NEXT: 0 |       class std::__shared_ptr<class sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT: 0 |         class std::__shared_ptr_access<class sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT: 0 |         element_type * _M_ptr
// CHECK-NEXT: 8 |         class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 8 |           _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 0 |   class sycl::detail::accessor_common<int, 1, sycl::access::mode::read, sycl::access::target::global_buffer, sycl::access::placeholder::false_t> (base) (empty)
// CHECK-NEXT: 0 |   class sycl::detail::OwnerLessBase<class sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::global_buffer, sycl::access::placeholder::false_t> > (base) (empty)
// CHECK-NEXT: 16 | detail::AccHostDataT * MAccData
// CHECK-NEXT: 24 |   char[8] padding
// CHECK-NEXT: [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: nvsize=32, nvalign=8]

//----------------------------------------------------------------------------//
// Local accessor.
//----------------------------------------------------------------------------//

void hostAcc(accessor<int, 1, access::mode::read_write, access::target::local> Acc) {
  (void)Acc.get_size();
}

// CHECK:      0 | class sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local, sycl::access::placeholder::false_t>
// CHECK-NEXT: 0 |   class sycl::local_accessor_base<int, 1, sycl::access::mode::read_write, sycl::access::placeholder::false_t> (base)
// CHECK-NEXT: 0 |     class sycl::detail::LocalAccessorBaseHost (base)
// CHECK-NEXT: 0 |       class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> impl
// CHECK-NEXT: 0 |         class std::__shared_ptr<class sycl::detail::LocalAccessorImplHost, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT: 0 |           class std::__shared_ptr_access<class sycl::detail::LocalAccessorImplHost, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT: 0 |           element_type * _M_ptr
// CHECK-NEXT: 8 |           class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 8 |             _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 0 |     class sycl::detail::accessor_common<int, 1, sycl::access::mode::read_write, sycl::access::target::local, sycl::access::placeholder::false_t> (base) (empty)
// CHECK-NEXT: 16 |     char[16] padding
// CHECK-NEXT: 0 |   class sycl::detail::OwnerLessBase<class sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local, sycl::access::placeholder::false_t> > (base) (empty)
// CHECK-NEXT: [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: nvsize=32, nvalign=8]

void hostAcc(local_accessor<int, 1> Acc) {
  (void)Acc.get_size();
}

// CHECK:      0 | class sycl::local_accessor<int, 1>
// CHECK-NEXT: 0 |   class sycl::local_accessor_base<int, 1, sycl::access::mode::read_write, sycl::access::placeholder::false_t> (base)
// CHECK-NEXT: 0 |     class sycl::detail::LocalAccessorBaseHost (base)
// CHECK-NEXT: 0 |       class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> impl
// CHECK-NEXT: 0 |         class std::__shared_ptr<class sycl::detail::LocalAccessorImplHost, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT: 0 |           class std::__shared_ptr_access<class sycl::detail::LocalAccessorImplHost, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT: 0 |           element_type * _M_ptr
// CHECK-NEXT: 8 |           class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 8 |             _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 0 |     class sycl::detail::accessor_common<int, 1, sycl::access::mode::read_write, sycl::access::target::local, sycl::access::placeholder::false_t> (base) (empty)
// CHECK-NEXT: 16 |     char[16] padding
// CHECK-NEXT: 0 |   class sycl::detail::OwnerLessBase<class sycl::local_accessor<int, 1> > (base) (empty)
// CHECK-NEXT: [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: nvsize=32, nvalign=8]

//----------------------------------------------------------------------------//
// Host image accessor.
//----------------------------------------------------------------------------//

void hostAcc(accessor<int4, 1, access::mode::read_write, access::target::host_image> Acc) {
  (void)Acc.get_count();
}
// CHECK:      0 | class sycl::accessor<class sycl::vec<int, 4>, 1, sycl::access::mode::read_write, sycl::access::target::host_image, sycl::access::placeholder::false_t>
// CHECK-NEXT: 0 |   class sycl::detail::image_accessor<class sycl::vec<int, 4>, 1, sycl::access::mode::read_write, sycl::access::target::host_image, sycl::access::placeholder::false_t> (base)
// CHECK-NEXT: 0 |     class sycl::detail::AccessorBaseHost (base)
// CHECK-NEXT: 0 |       class std::shared_ptr<class sycl::detail::AccessorImplHost> impl
// CHECK-NEXT: 0 |         class std::__shared_ptr<class sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT: 0 |           class std::__shared_ptr_access<class sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT: 0 |           element_type * _M_ptr
// CHECK-NEXT: 8 |           class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 8 |             _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 16 |     size_t MImageCount
// CHECK-NEXT: 24 |     image_channel_order MImgChannelOrder
// CHECK-NEXT: 28 |     image_channel_type MImgChannelType
// CHECK-NEXT: 0 |   class sycl::detail::OwnerLessBase<class sycl::accessor<class sycl::vec<int, 4>, 1, sycl::access::mode::read_write, sycl::access::target::host_image, sycl::access::placeholder::false_t> > (base) (empty)
// CHECK-NEXT: [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: nvsize=32, nvalign=8]

//----------------------------------------------------------------------------//
// Image accessor.
//----------------------------------------------------------------------------//

void hostAcc(accessor<int4, 1, access::mode::read, access::target::image> Acc) {
  (void)Acc.get_count();
}
// CHECK:      0 | class sycl::accessor<class sycl::vec<int, 4>, 1, sycl::access::mode::read, sycl::access::target::image, sycl::access::placeholder::false_t>
// CHECK-NEXT: 0 |   class sycl::detail::image_accessor<class sycl::vec<int, 4>, 1, sycl::access::mode::read, sycl::access::target::image, sycl::access::placeholder::false_t> (base)
// CHECK-NEXT: 0 |     class sycl::detail::AccessorBaseHost (base)
// CHECK-NEXT: 0 |       class std::shared_ptr<class sycl::detail::AccessorImplHost> impl
// CHECK-NEXT: 0 |         class std::__shared_ptr<class sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT: 0 |           class std::__shared_ptr_access<class sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT: 0 |           element_type * _M_ptr
// CHECK-NEXT: 8 |           class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 8 |             _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 16 |     size_t MImageCount
// CHECK-NEXT: 24 |     image_channel_order MImgChannelOrder
// CHECK-NEXT: 28 |     image_channel_type MImgChannelType
// CHECK-NEXT: 0 |   class sycl::detail::OwnerLessBase<class sycl::accessor<class sycl::vec<int, 4>, 1, sycl::access::mode::read, sycl::access::target::image, sycl::access::placeholder::false_t> > (base) (empty)
// CHECK-NEXT: [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: nvsize=32, nvalign=8]
