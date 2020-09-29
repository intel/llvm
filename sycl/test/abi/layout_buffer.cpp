// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s | FileCheck %s
// REQUIRES: linux

// clang-format off

#include <CL/sycl/buffer.hpp>

void foo(sycl::buffer<int, 2>) {}
// CHECK: 0 | class cl::sycl::detail::buffer_impl
// CHEK-NEXT: 0 |   class cl::sycl::detail::SYCLMemObjT (primary base)
// CHEK-NEXT:  0 |     class cl::sycl::detail::SYCLMemObjI (primary base)
// CHEK-NEXT:  0 |       (SYCLMemObjI vtable pointer)
// CHEK-NEXT:  8 |       class std::shared_ptr<struct cl::sycl::detail::MemObjRecord> MRecord
// CHEK-NEXT:  8 |         class std::__shared_ptr<struct cl::sycl::detail::MemObjRecord, __gnu_cxx::_S_atomic> (base)
// CHEK-NEXT:  8 |           class std::__shared_ptr_access<struct cl::sycl::detail::MemObjRecord, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHEK-NEXT:  8 |           std::__shared_ptr<struct cl::sycl::detail::MemObjRecord, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHEK-NEXT:  16 |           class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHEK-NEXT:  16 |             _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHEK-NEXT:  24 |     class std::unique_ptr<class cl::sycl::detail::SYCLMemObjAllocator, struct std::default_delete<class cl::sycl::detail::SYCLMemObjAllocator> > MAllocator
// CHEK-NEXT:  24 |       struct std::__uniq_ptr_data<class cl::sycl::detail::SYCLMemObjAllocator, struct std::default_delete<class cl::sycl::detail::SYCLMemObjAllocator>, true, true> _M_t
// CHEK-NEXT:  24 |         class std::__uniq_ptr_impl<class cl::sycl::detail::SYCLMemObjAllocator, struct std::default_delete<class cl::sycl::detail::SYCLMemObjAllocator> > (base)
// CHEK-NEXT:  24 |           class std::tuple<class cl::sycl::detail::SYCLMemObjAllocator *, struct std::default_delete<class cl::sycl::detail::SYCLMemObjAllocator> > _M_t
// CHEK-NEXT:  24 |             struct std::_Tuple_impl<0, class cl::sycl::detail::SYCLMemObjAllocator *, struct std::default_delete<class cl::sycl::detail::SYCLMemObjAllocator> > (base)
// CHEK-NEXT:  24 |               struct std::_Tuple_impl<1, struct std::default_delete<class cl::sycl::detail::SYCLMemObjAllocator> > (base) (empty)
// CHEK-NEXT:  24 |                 struct std::_Head_base<1, struct std::default_delete<class cl::sycl::detail::SYCLMemObjAllocator>, true> (base) (empty)
// CHEK-NEXT:  24 |                   struct std::default_delete<class cl::sycl::detail::SYCLMemObjAllocator> (base) (empty)
// CHEK-NEXT:  24 |               struct std::_Head_base<0, class cl::sycl::detail::SYCLMemObjAllocator *, false> (base)
// CHEK-NEXT:  24 |                 class cl::sycl::detail::SYCLMemObjAllocator * _M_head_impl
// CHEK-NEXT:  32 |     class cl::sycl::property_list MProps
// CHEK-NEXT:  32 |       class cl::sycl::detail::PropertyListBase (base)
// CHEK-NEXT:  32 |         class std::bitset<7> MDataLessProps
// CHEK-NEXT:  32 |           struct std::_Base_bitset<1> (base)
// CHEK-NEXT:  32 |             std::_Base_bitset<1>::_WordT _M_w
// CHEK-NEXT:  40 |         class std::vector<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase> > > MPropsWithData
// CHEK-NEXT:  40 |           struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase> > > (base)
// CHEK-NEXT:  40 |             struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase> > >::_Vector_impl _M_impl
// CHEK-NEXT:  40 |               class std::allocator<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase> > (base) (empty)
// CHEK-NEXT:  40 |                 class __gnu_cxx::new_allocator<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase> > (base) (empty)
// CHEK-NEXT:  40 |               struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase> > >::_Vector_impl_data (base)
// CHEK-NEXT:  40 |                 std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase> > >::pointer _M_start
// CHEK-NEXT:  48 |                 std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase> > >::pointer _M_finish
// CHEK-NEXT:  56 |                 std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::PropertyWithDataBase> > >::pointer _M_end_of_storage
// CHEK-NEXT:  64 |     class std::shared_ptr<class cl::sycl::detail::event_impl> MInteropEvent
// CHEK-NEXT:  64 |       class std::__shared_ptr<class cl::sycl::detail::event_impl, __gnu_cxx::_S_atomic> (base)
// CHEK-NEXT:  64 |         class std::__shared_ptr_access<class cl::sycl::detail::event_impl, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHEK-NEXT:  64 |         std::__shared_ptr<class cl::sycl::detail::event_impl, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHEK-NEXT:  72 |         class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHEK-NEXT:  72 |           _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHEK-NEXT:  80 |     class std::shared_ptr<class cl::sycl::detail::context_impl> MInteropContext
// CHEK-NEXT:  80 |       class std::__shared_ptr<class cl::sycl::detail::context_impl, __gnu_cxx::_S_atomic> (base)
// CHEK-NEXT:  80 |         class std::__shared_ptr_access<class cl::sycl::detail::context_impl, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHEK-NEXT:  80 |         std::__shared_ptr<class cl::sycl::detail::context_impl, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHEK-NEXT:  88 |         class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHEK-NEXT:  88 |           _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHEK-NEXT:  96 |     cl_mem MInteropMemObject
// CHEK-NEXT:  104 |     _Bool MOpenCLInterop
// CHEK-NEXT:  105 |     _Bool MHostPtrReadOnly
// CHEK-NEXT:  106 |     _Bool MNeedWriteBack
// CHEK-NEXT:  112 |     size_t MSizeInBytes
// CHEK-NEXT:  120 |     void * MUserPtr
// CHEK-NEXT:  128 |     void * MShadowCopy
// CHEK-NEXT:  136 |     class std::function<void (void)> MUploadDataFunctor
// CHEK-NEXT:  136 |       struct std::_Maybe_unary_or_binary_function<void> (base) (empty)
// CHEK-NEXT:  136 |       class std::_Function_base (base)
// CHEK-NEXT:  136 |         union std::_Any_data _M_functor
// CHEK-NEXT:  136 |           union std::_Nocopy_types _M_unused
// CHEK-NEXT:  136 |             void * _M_object
// CHEK-NEXT:  136 |             const void * _M_const_object
// CHEK-NEXT:  136 |             void (*)(void) _M_function_pointer
// CHEK-NEXT:  136 |             void (class std::_Undefined_class::*)(void) _M_member_pointer
// CHEK-NEXT:  136 |           char [16] _M_pod_data
// CHEK-NEXT:  152 |         std::_Function_base::_Manager_type _M_manager
// CHEK-NEXT:  160 |       std::function<void (void)>::_Invoker_type _M_invoker
// CHEK-NEXT:  168 |     class std::shared_ptr<const void> MSharedPtrStorage
// CHEK-NEXT:  168 |       class std::__shared_ptr<const void, __gnu_cxx::_S_atomic> (base)
// CHEK-NEXT:  168 |         class std::__shared_ptr_access<const void, __gnu_cxx::_S_atomic, false, true> (base) (empty)
// CHEK-NEXT:  168 |         std::__shared_ptr<const void, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHEK-NEXT:  176 |         class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHEK-NEXT:  176 |           _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHEK-NEXT:      | [sizeof=184, dsize=184, align=8,
// CHEK-NEXT:      |  nvsize=184, nvalign=8]

// CHEK: 0 | class cl::sycl::buffer<int, 2, class cl::sycl::detail::aligned_allocator<char>, void>
// CHEK-NEXT: 0 |   class std::shared_ptr<class cl::sycl::detail::buffer_impl> impl
// CHEK-NEXT: 0 |     class std::__shared_ptr<class cl::sycl::detail::buffer_impl, __gnu_cxx::_S_atomic> (base)
// CHEK-NEXT: 0 |       class std::__shared_ptr_access<class cl::sycl::detail::buffer_impl, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHEK-NEXT: 0 |       std::__shared_ptr<class cl::sycl::detail::buffer_impl, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHEK-NEXT: 8 |       class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHEK-NEXT: 8 |         _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHEK-NEXT: 16 |   class cl::sycl::range<2> Range
// CHEK-NEXT: 16 |     class cl::sycl::detail::array<2> (base)
// CHEK-NEXT: 16 |       size_t [2] common_array
// CHEK-NEXT: 32 |   size_t OffsetInBytes
// CHEK-NEXT: 40 |   _Bool IsSubBuffer
// CHEK-NEXT:    | [sizeof=48, dsize=41, align=8,
// CHEK-NEXT:    |  nvsize=41, nvalign=8]
