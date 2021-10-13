// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <CL/sycl/buffer.hpp>

void foo(sycl::buffer<int, 2>) {}
// CHECK: 0 | class sycl::detail::buffer_impl
// CHECK-NEXT: 0 |   class sycl::detail::SYCLMemObjT (primary base)
// CHECK-NEXT:  0 |     class sycl::detail::SYCLMemObjI (primary base)
// CHECK-NEXT:  0 |       (SYCLMemObjI vtable pointer)
// CHECK-NEXT:  8 |       class std::shared_ptr<struct sycl::detail::MemObjRecord> MRecord
// CHECK-NEXT:  8 |         class std::__shared_ptr<struct sycl::detail::MemObjRecord, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT:  8 |           class std::__shared_ptr_access<struct sycl::detail::MemObjRecord, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT:  8 |           std::__shared_ptr<struct sycl::detail::MemObjRecord, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT:  16 |           class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT:  16 |             _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT:  24 |     class std::unique_ptr<class sycl::detail::SYCLMemObjAllocator> MAllocator
// CHECK:  24 |         class std::__uniq_ptr_impl<class sycl::detail::SYCLMemObjAllocator, struct std::default_delete<class sycl::detail::SYCLMemObjAllocator> >
// CHECK-NEXT:  24 |           class std::tuple<class sycl::detail::SYCLMemObjAllocator *, struct std::default_delete<class sycl::detail::SYCLMemObjAllocator> > _M_t
// CHECK-NEXT:  24 |             struct std::_Tuple_impl<0, class sycl::detail::SYCLMemObjAllocator *, struct std::default_delete<class sycl::detail::SYCLMemObjAllocator> > (base)
// CHECK-NEXT:  24 |               struct std::_Tuple_impl<1, struct std::default_delete<class sycl::detail::SYCLMemObjAllocator> > (base) (empty)
// CHECK-NEXT:  24 |                 struct std::_Head_base<1, struct std::default_delete<class sycl::detail::SYCLMemObjAllocator>, true> (base) (empty)
// CHECK-NEXT:  24 |                   struct std::default_delete<class sycl::detail::SYCLMemObjAllocator>
// CHECK-NEXT:  24 |               struct std::_Head_base<0, class sycl::detail::SYCLMemObjAllocator *, false> (base)
// CHECK-NEXT:  24 |                 class sycl::detail::SYCLMemObjAllocator * _M_head_impl
// CHECK-NEXT:  32 |     class sycl::property_list MProps
// CHECK-NEXT:  32 |       class sycl::detail::PropertyListBase (base)
// CHECK-NEXT:  32 |         class std::bitset<32> MDataLessProps
// CHECK-NEXT:  32 |           struct std::_Base_bitset<1> (base)
// CHECK-NEXT:  32 |             std::_Base_bitset<1>::_WordT _M_w
// CHECK-NEXT:  40 |         class std::vector<class std::shared_ptr<class sycl::detail::PropertyWithDataBase> > MPropsWithData
// CHECK-NEXT:  40 |           struct std::_Vector_base<class std::shared_ptr<class sycl::detail::PropertyWithDataBase>, class std::allocator<class std::shared_ptr<class sycl::detail::PropertyWithDataBase> > > (base)
// CHECK-NEXT:  40 |             struct std::_Vector_base<class std::shared_ptr<class sycl::detail::PropertyWithDataBase>, class std::allocator<class std::shared_ptr<class sycl::detail::PropertyWithDataBase> > >::_Vector_impl _M_impl
// CHECK-NEXT:  40 |               class std::allocator<class std::shared_ptr<class sycl::detail::PropertyWithDataBase> > (base) (empty)
// CHECK-NEXT:  40 |                 class __gnu_cxx::new_allocator<class std::shared_ptr<class sycl::detail::PropertyWithDataBase> > (base) (empty)
// CHECK:  40 |                 std::_Vector_base<class std::shared_ptr<class sycl::detail::PropertyWithDataBase>, class std::allocator<class std::shared_ptr<class sycl::detail::PropertyWithDataBase> > >::pointer _M_start
// CHECK-NEXT:  48 |                 std::_Vector_base<class std::shared_ptr<class sycl::detail::PropertyWithDataBase>, class std::allocator<class std::shared_ptr<class sycl::detail::PropertyWithDataBase> > >::pointer _M_finish
// CHECK-NEXT:  56 |                 std::_Vector_base<class std::shared_ptr<class sycl::detail::PropertyWithDataBase>, class std::allocator<class std::shared_ptr<class sycl::detail::PropertyWithDataBase> > >::pointer _M_end_of_storage
// CHECK-NEXT:  64 |     class std::shared_ptr<class sycl::detail::event_impl> MInteropEvent
// CHECK-NEXT:  64 |       class std::__shared_ptr<class sycl::detail::event_impl, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT:  64 |         class std::__shared_ptr_access<class sycl::detail::event_impl, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT:  64 |         std::__shared_ptr<class sycl::detail::event_impl, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT:  72 |         class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT:  72 |           _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT:  80 |     class std::shared_ptr<class sycl::detail::context_impl> MInteropContext
// CHECK-NEXT:  80 |       class std::__shared_ptr<class sycl::detail::context_impl, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT:  80 |         class std::__shared_ptr_access<class sycl::detail::context_impl, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT:  80 |         std::__shared_ptr<class sycl::detail::context_impl, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT:  88 |         class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT:  88 |           _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT:  96 |     cl_mem MInteropMemObject
// CHECK-NEXT:  104 |     _Bool MOpenCLInterop
// CHECK-NEXT:  105 |     _Bool MHostPtrReadOnly
// CHECK-NEXT:  106 |     _Bool MNeedWriteBack
// CHECK-NEXT:  112 |     size_t MSizeInBytes
// CHECK-NEXT:  120 |     void * MUserPtr
// CHECK-NEXT:  128 |     void * MShadowCopy
// CHECK-NEXT:  136 |     class std::function<void (void)> MUploadDataFunctor
// CHECK-NEXT:  136 |       struct std::_Maybe_unary_or_binary_function<void> (base) (empty)
// CHECK-NEXT:  136 |       class std::_Function_base (base)
// CHECK-NEXT:  136 |         union std::_Any_data _M_functor
// CHECK-NEXT:  136 |           union std::_Nocopy_types _M_unused
// CHECK-NEXT:  136 |             void * _M_object
// CHECK-NEXT:  136 |             const void * _M_const_object
// CHECK-NEXT:  136 |             void (*)(void) _M_function_pointer
// CHECK-NEXT:  136 |             void (class std::_Undefined_class::*)(void) _M_member_pointer
// CHECK-NEXT:  136 |           char [16] _M_pod_data
// CHECK-NEXT:  152 |         std::_Function_base::_Manager_type _M_manager
// CHECK-NEXT:  160 |       std::function<void (void)>::_Invoker_type _M_invoker
// CHECK-NEXT:  168 |     class std::shared_ptr<const void> MSharedPtrStorage
// CHECK-NEXT:  168 |       class std::__shared_ptr<const void, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT:  168 |         class std::__shared_ptr_access<const void, __gnu_cxx::_S_atomic, false, true> (base) (empty)
// CHECK-NEXT:  168 |         std::__shared_ptr<const void, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT:  176 |         class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT:  176 |           _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT:      | [sizeof=184, dsize=184, align=8,
// CHECK-NEXT:      |  nvsize=184, nvalign=8]

// CHECK: 0 | class sycl::buffer<int, 2, class sycl::detail::aligned_allocator<char>, void>
// CHECK-NEXT: 0 |   class std::shared_ptr<class sycl::detail::buffer_impl> impl
// CHECK-NEXT: 0 |     class std::__shared_ptr<class sycl::detail::buffer_impl, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT: 0 |       class std::__shared_ptr_access<class sycl::detail::buffer_impl, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT: 0 |       std::__shared_ptr<class sycl::detail::buffer_impl, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT: 8 |       class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 8 |         _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 16 |   class sycl::range<2> Range
// CHECK-NEXT: 16 |     class sycl::detail::array<2> (base)
// CHECK-NEXT: 16 |       size_t [2] common_array
// CHECK-NEXT: 32 |   size_t OffsetInBytes
// CHECK-NEXT: 40 |   _Bool IsSubBuffer
// CHECK-NEXT:    | [sizeof=48, dsize=41, align=8,
// CHECK-NEXT:    |  nvsize=41, nvalign=8]
