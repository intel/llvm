// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <CL/sycl/accessor.hpp>

using namespace cl::sycl;

// CHECK:        0 | class sycl::detail::AccessorImplHost
// CHECK-NEXT:   0 |   class sycl::id<3> MOffset
// CHECK-NEXT:   0 |     class sycl::detail::array<3> (base)
// CHECK-NEXT:   0 |       size_t [3] common_array
// CHECK-NEXT:  24 |   class sycl::range<3> MAccessRange
// CHECK-NEXT:  24 |     class sycl::detail::array<3> (base)
// CHECK-NEXT:  24 |       size_t [3] common_array
// CHECK-NEXT:  48 |   class sycl::range<3> MMemoryRange
// CHECK-NEXT:  48 |     class sycl::detail::array<3> (base)
// CHECK-NEXT:  48 |       size_t [3] common_array
// CHECK-NEXT:  72 |   access::mode MAccessMode
// CHECK-NEXT:  80 |   detail::SYCLMemObjI * MSYCLMemObj
// CHECK-NEXT:  88 |   unsigned int MDims
// CHECK-NEXT:  92 |   unsigned int MElemSize
// CHECK-NEXT:  96 |   unsigned int MOffsetInBytes
// CHECK-NEXT: 100 |   _Bool MIsSubBuffer
// CHECK-NEXT: 104 |   void * MData
// CHECK-NEXT: 112 |   class sycl::detail::Command * MBlockedCmd
// CHECK-NEXT: 120 |   _Bool PerWI
// CHECK-NEXT: 121 |   _Bool MIsESIMDAcc
// CHECK-NEXT:     | [sizeof=128, dsize=122, align=8,
// CHECK-NEXT:     |  nvsize=122, nvalign=8]

// CHECK:       0 | class sycl::detail::LocalAccessorImplHost
// CHECK-NEXT:  0 |   class sycl::range<3> MSize
// CHECK-NEXT:  0 |     class sycl::detail::array<3> (base)
// CHECK-NEXT:  0 |       size_t [3] common_array
// CHECK-NEXT: 24 |   int MDims
// CHECK-NEXT: 28 |   int MElemSize
// CHECK-NEXT: 32 |   class std::vector<char> MMem
// CHECK-NEXT: 32 |     struct std::_Vector_base<char, class std::allocator<char> > (base)
// CHECK-NEXT: 32 |       struct std::_Vector_base<char, class std::allocator<char> >::_Vector_impl _M_impl
// CHECK-NEXT: 32 |         class std::allocator<char> (base) (empty)
// CHECK-NEXT: 32 |           class __gnu_cxx::new_allocator<char> (base) (empty)
// CHECK:      32 |           std::_Vector_base<char, class std::allocator<char> >::pointer _M_start
// CHECK-NEXT: 40 |           std::_Vector_base<char, class std::allocator<char> >::pointer _M_finish
// CHECK-NEXT: 48 |           std::_Vector_base<char, class std::allocator<char> >::pointer _M_end_of_storage
// CHECK-NEXT:    | [sizeof=56, dsize=56, align=8,
// CHECK-NEXT:    |  nvsize=56, nvalign=8]

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
// CHECK-NEXT:  0 |         std::__shared_ptr<class sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT:  8 |         class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT:  8 |           _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT:  0 |   class sycl::detail::accessor_common<int, 1, sycl::access::mode::read, sycl::access::target::host_buffer, sycl::access::placeholder::false_t> (base) (empty)
// CHECK-NEXT: 16 |   char [16] padding
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
// CHECK-NEXT: 0 |         std::__shared_ptr<class sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT: 8 |         class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 8 |           _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 0 |   class sycl::detail::accessor_common<int, 1, sycl::access::mode::read, sycl::access::target::global_buffer, sycl::access::placeholder::false_t> (base) (empty)
// CHECK-NEXT: 16 |   char [16] padding
// CHECK-NEXT: [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: nvsize=32, nvalign=8]

//----------------------------------------------------------------------------//
// Local accessor.
//----------------------------------------------------------------------------//

void hostAcc(accessor<int, 1, access::mode::read_write, access::target::local> Acc) {
  (void)Acc.get_size();
}

// CHECK: 0      | class sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local, sycl::access::placeholder::false_t>
// CHECK-NEXT: 0 |   class sycl::detail::LocalAccessorBaseHost (base)
// CHECK-NEXT: 0 |     class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> impl
// CHECK-NEXT: 0 |       class std::__shared_ptr<class sycl::detail::LocalAccessorImplHost, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT: 0 |         class std::__shared_ptr_access<class sycl::detail::LocalAccessorImplHost, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT: 0 |         std::__shared_ptr<class sycl::detail::LocalAccessorImplHost, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT: 8 |         class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 8 |           _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 0 |   class sycl::detail::accessor_common<int, 1, sycl::access::mode::read_write, sycl::access::target::local, sycl::access::placeholder::false_t> (base) (empty)
// CHECK-NEXT: 16 |   char [16] padding
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
// CHECK-NEXT: 0 |           std::__shared_ptr<class sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT: 8 |           class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 8 |             _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 16 |     size_t MImageCount
// CHECK-NEXT: 24 |     enum sycl::image_channel_order MImgChannelOrder
// CHECK-NEXT: 28 |     enum sycl::image_channel_type MImgChannelType
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
// CHECK-NEXT: 0 |           std::__shared_ptr<class sycl::detail::AccessorImplHost, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT: 8 |           class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 8 |             _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 16 |     size_t MImageCount
// CHECK-NEXT: 24 |     enum sycl::image_channel_order MImgChannelOrder
// CHECK-NEXT: 28 |     enum sycl::image_channel_type MImgChannelType
// CHECK-NEXT: [sizeof=32, dsize=32, align=8,
// CHECK-NEXT: nvsize=32, nvalign=8]
