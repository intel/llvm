// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s | FileCheck %s
// REQUIRES: linux

// clang-format off

#include <CL/sycl/image.hpp>

sycl::image<2> Img{sycl::image_channel_order::rgba, sycl::image_channel_type::fp16, sycl::range<2>{10, 10}};

// CHECK: 0 | class cl::sycl::detail::image_impl<2>
// CHECK-NEXT: 0 |   class cl::sycl::detail::SYCLMemObjT (primary base)
// CHECK-NEXT: 0 |     class cl::sycl::detail::SYCLMemObjI (primary base)
// CHECK-NEXT: 0 |       (SYCLMemObjI vtable pointer)
// CHECK-NEXT: 8 |       class std::shared_ptr<struct cl::sycl::detail::MemObjRecord> MRecord
// CHECK-NEXT: 8 |         class std::__shared_ptr<struct cl::sycl::detail::MemObjRecord, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT: 8 |           class std::__shared_ptr_access<struct cl::sycl::detail::MemObjRecord, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT: 8 |           std::__shared_ptr<struct cl::sycl::detail::MemObjRecord, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT: 16 |           class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 16 |             _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 24 |     class std::unique_ptr<class cl::sycl::detail::SYCLMemObjAllocator, struct std::default_delete<class cl::sycl::detail::SYCLMemObjAllocator> > MAllocator
// CHECK: 24 |         class std::__uniq_ptr_impl<class cl::sycl::detail::SYCLMemObjAllocator, struct std::default_delete<class cl::sycl::detail::SYCLMemObjAllocator> >
// CHECK-NEXT: 24 |           class std::tuple<class cl::sycl::detail::SYCLMemObjAllocator *, struct std::default_delete<class cl::sycl::detail::SYCLMemObjAllocator> > _M_t
// CHECK-NEXT: 24 |             struct std::_Tuple_impl<0, class cl::sycl::detail::SYCLMemObjAllocator *, struct std::default_delete<class cl::sycl::detail::SYCLMemObjAllocator> > (base)
// CHECK-NEXT: 24 |               struct std::_Tuple_impl<1, struct std::default_delete<class cl::sycl::detail::SYCLMemObjAllocator> > (base) (empty)
// CHECK-NEXT: 24 |                 struct std::_Head_base<1, struct std::default_delete<class cl::sycl::detail::SYCLMemObjAllocator>, true> (base) (empty)
// CHECK-NEXT: 24 |                   struct std::default_delete<class cl::sycl::detail::SYCLMemObjAllocator> (base) (empty)
// CHECK-NEXT: 24 |               struct std::_Head_base<0, class cl::sycl::detail::SYCLMemObjAllocator *, false> (base)
// CHECK-NEXT: 24 |                 class cl::sycl::detail::SYCLMemObjAllocator * _M_head_impl
// CHECK-NEXT: 32 |     class cl::sycl::property_list MProps
// CHECK-NEXT: 32 |       class cl::sycl::detail::PropertyListBase (base)
// CHECK-NEXT: 32 |         class std::bitset<7> MDataLessProps
// CHECK-NEXT: 32 |           struct std::_Base_bitset<1> (base)
// CHECK-NEXT: 32 |             std::_Base_bitset<1>::_WordT _M_w
// CHECK-NEXT: 40 |         class std::vector<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase> > > MPropsWithData
// CHECK-NEXT: 40 |           struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase> > > (base)
// CHECK-NEXT: 40 |             struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase> > >::_Vector_impl _M_impl
// CHECK-NEXT: 40 |               class std::allocator<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase> > (base) (empty)
// CHECK-NEXT: 40 |                 class __gnu_cxx::new_allocator<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase> > (base) (empty)
// CHECK: 40 |                 std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase> > >::pointer _M_start
// CHECK-NEXT: 48 |                 std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase> > >::pointer _M_finish
// CHECK-NEXT: 56 |                 std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase> > >::pointer _M_end_of_storage
// CHECK-NEXT: 64 |     class std::shared_ptr<class cl::sycl::detail::event_impl> MInteropEvent
// CHECK-NEXT: 64 |       class std::__shared_ptr<class cl::sycl::detail::event_impl, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT: 64 |         class std::__shared_ptr_access<class cl::sycl::detail::event_impl, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT: 64 |         std::__shared_ptr<class cl::sycl::detail::event_impl, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT: 72 |         class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 72 |           _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 80 |     class std::shared_ptr<class cl::sycl::detail::context_impl> MInteropContext
// CHECK-NEXT: 80 |       class std::__shared_ptr<class cl::sycl::detail::context_impl, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT: 80 |         class std::__shared_ptr_access<class cl::sycl::detail::context_impl, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT: 80 |         std::__shared_ptr<class cl::sycl::detail::context_impl, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT: 88 |         class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 88 |           _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 96 |     cl_mem MInteropMemObject
// CHECK-NEXT: 104 |     _Bool MOpenCLInterop
// CHECK-NEXT: 105 |     _Bool MHostPtrReadOnly
// CHECK-NEXT: 106 |     _Bool MNeedWriteBack
// CHECK-NEXT: 112 |     size_t MSizeInBytes
// CHECK-NEXT: 120 |     void * MUserPtr
// CHECK-NEXT: 128 |     void * MShadowCopy
// CHECK-NEXT: 136 |     class std::function<void (void)> MUploadDataFunctor
// CHECK-NEXT: 136 |       struct std::_Maybe_unary_or_binary_function<void> (base) (empty)
// CHECK-NEXT: 136 |       class std::_Function_base (base)
// CHECK-NEXT: 136 |         union std::_Any_data _M_functor
// CHECK-NEXT: 136 |           union std::_Nocopy_types _M_unused
// CHECK-NEXT: 136 |             void * _M_object
// CHECK-NEXT: 136 |             const void * _M_const_object
// CHECK-NEXT: 136 |             void (*)(void) _M_function_pointer
// CHECK-NEXT: 136 |             void (class std::_Undefined_class::*)(void) _M_member_pointer
// CHECK-NEXT: 136 |           char [16] _M_pod_data
// CHECK-NEXT: 152 |         std::_Function_base::_Manager_type _M_manager
// CHECK-NEXT: 160 |       std::function<void (void)>::_Invoker_type _M_invoker
// CHECK-NEXT: 168 |     class std::shared_ptr<const void> MSharedPtrStorage
// CHECK-NEXT: 168 |       class std::__shared_ptr<const void, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT: 168 |         class std::__shared_ptr_access<const void, __gnu_cxx::_S_atomic, false, true> (base) (empty)
// CHECK-NEXT: 168 |         std::__shared_ptr<const void, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT: 176 |         class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 176 |           _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 184 |   _Bool MIsArrayImage
// CHECK-NEXT: 192 |   class cl::sycl::range<2> MRange
// CHECK-NEXT: 192 |     class cl::sycl::detail::array<2> (base)
// CHECK-NEXT: 192 |       size_t [2] common_array
// CHECK-NEXT: 208 |   enum cl::sycl::image_channel_order MOrder
// CHECK-NEXT: 212 |   enum cl::sycl::image_channel_type MType
// CHECK-NEXT: 216 |   uint8_t MNumChannels
// CHECK-NEXT: 224 |   size_t MElementSize
// CHECK-NEXT: 232 |   size_t MRowPitch
// CHECK-NEXT: 240 |   size_t MSlicePitch
// CHECK-NEXT:     | [sizeof=248, dsize=248, align=8,
// CHECK-NEXT:     |  nvsize=248, nvalign=8]

// CHECK: 0 | class cl::sycl::image<2, class cl::sycl::detail::aligned_allocator<unsigned char> >
// CHECK-NEXT: 0 |   class std::shared_ptr<class cl::sycl::detail::image_impl<2> > impl
// CHECK-NEXT: 0 |     class std::__shared_ptr<class cl::sycl::detail::image_impl<2>, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT: 0 |       class std::__shared_ptr_access<class cl::sycl::detail::image_impl<2>, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT: 0 |       std::__shared_ptr<class cl::sycl::detail::image_impl<2>, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT: 8 |       class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 8 |         _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT:   | [sizeof=16, dsize=16, align=8,
// CHECK-NEXT:   |  nvsize=16, nvalign=8]
