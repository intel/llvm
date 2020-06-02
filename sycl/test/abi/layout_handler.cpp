// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s | FileCheck %s
// REQUIRES: linux

#include <CL/sycl/handler.hpp>
#include <CL/sycl/queue.hpp>

void foo() {
  sycl::queue Q;
  Q.submit([](sycl::handler &CGH) {
    CGH.single_task<class Test>([]() {});
  });
}

// The order of field declarations and their types are important.

// CHECK: 0 | class cl::sycl::handler
// CHECK-NEXT: 0 |   class std::shared_ptr<class cl::sycl::detail::queue_impl> MQueue
// CHECK-NEXT: 0 |     class std::__shared_ptr<class cl::sycl::detail::queue_impl, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT: 0 |       class std::__shared_ptr_access<class cl::sycl::detail::queue_impl, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT: 0 |       std::__shared_ptr<class cl::sycl::detail::queue_impl, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT: 8 |       class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 8 |         _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 16 |   class std::vector<class std::vector<char, class std::allocator<char> >, class std::allocator<class std::vector<char, class std::allocator<char> > > > MArgsStorage
// CHECK-NEXT: 16 |     struct std::_Vector_base<class std::vector<char, class std::allocator<char> >, class std::allocator<class std::vector<char, class std::allocator<char> > > > (base)
// CHECK-NEXT: 16 |       struct std::_Vector_base<class std::vector<char, class std::allocator<char> >, class std::allocator<class std::vector<char, class std::allocator<char> > > >::_Vector_impl _M_impl
// CHECK-NEXT: 16 |         class std::allocator<class std::vector<char, class std::allocator<char> > > (base) (empty)
// CHECK-NEXT: 16 |           class __gnu_cxx::new_allocator<class std::vector<char, class std::allocator<char> > > (base) (empty)
// CHECK-NEXT: 16 |         std::_Vector_base<class std::vector<char, class std::allocator<char> >, class std::allocator<class std::vector<char, class std::allocator<char> > > >::pointer _M_start
// CHECK-NEXT: 24 |         std::_Vector_base<class std::vector<char, class std::allocator<char> >, class std::allocator<class std::vector<char, class std::allocator<char> > > >::pointer _M_finish
// CHECK-NEXT: 32 |         std::_Vector_base<class std::vector<char, class std::allocator<char> >, class std::allocator<class std::vector<char, class std::allocator<char> > > >::pointer _M_end_of_storage
// CHECK-NEXT: 40 |   class std::vector<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> > > MAccStorage
// CHECK-NEXT: 40 |     struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> > > (base)
// CHECK-NEXT: 40 |       struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> > >::_Vector_impl _M_impl
// CHECK-NEXT: 40 |         class std::allocator<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> > (base) (empty)
// CHECK-NEXT: 40 |           class __gnu_cxx::new_allocator<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> > (base) (empty)
// CHECK-NEXT: 40 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> > >::pointer _M_start
// CHECK-NEXT: 48 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> > >::pointer _M_finish
// CHECK-NEXT: 56 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> > >::pointer _M_end_of_storage
// CHECK-NEXT: 64 |   class std::vector<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost> > > MLocalAccStorage
// CHECK-NEXT: 64 |     struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost> > > (base)
// CHECK-NEXT: 64 |       struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost> > >::_Vector_impl _M_impl
// CHECK-NEXT: 64 |         class std::allocator<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost> > (base) (empty)
// CHECK-NEXT: 64 |           class __gnu_cxx::new_allocator<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost> > (base) (empty)
// CHECK-NEXT: 64 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost> > >::pointer _M_start
// CHECK-NEXT: 72 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost> > >::pointer _M_finish
// CHECK-NEXT: 80 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost> > >::pointer _M_end_of_storage
// CHECK-NEXT: 88 |   class std::vector<class std::shared_ptr<class cl::sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::stream_impl> > > MStreamStorage
// CHECK-NEXT: 88 |     struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::stream_impl> > > (base)
// CHECK-NEXT: 88 |       struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::stream_impl> > >::_Vector_impl _M_impl
// CHECK-NEXT: 88 |         class std::allocator<class std::shared_ptr<class cl::sycl::detail::stream_impl> > (base) (empty)
// CHECK-NEXT: 88 |           class __gnu_cxx::new_allocator<class std::shared_ptr<class cl::sycl::detail::stream_impl> > (base) (empty)
// CHECK-NEXT: 88 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::stream_impl> > >::pointer _M_start
// CHECK-NEXT: 96 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::stream_impl> > >::pointer _M_finish
// CHECK-NEXT: 104 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::stream_impl> > >::pointer _M_end_of_storage
// CHECK-NEXT: 112 |   class std::vector<class std::shared_ptr<const void>, class std::allocator<class std::shared_ptr<const void> > > MSharedPtrStorage
// CHECK-NEXT: 112 |     struct std::_Vector_base<class std::shared_ptr<const void>, class std::allocator<class std::shared_ptr<const void> > > (base)
// CHECK-NEXT: 112 |       struct std::_Vector_base<class std::shared_ptr<const void>, class std::allocator<class std::shared_ptr<const void> > >::_Vector_impl _M_impl
// CHECK-NEXT: 112 |         class std::allocator<class std::shared_ptr<const void> > (base) (empty)
// CHECK-NEXT: 112 |           class __gnu_cxx::new_allocator<class std::shared_ptr<const void> > (base) (empty)
// CHECK-NEXT: 112 |         std::_Vector_base<class std::shared_ptr<const void>, class std::allocator<class std::shared_ptr<const void> > >::pointer _M_start
// CHECK-NEXT: 120 |         std::_Vector_base<class std::shared_ptr<const void>, class std::allocator<class std::shared_ptr<const void> > >::pointer _M_finish
// CHECK-NEXT: 128 |         std::_Vector_base<class std::shared_ptr<const void>, class std::allocator<class std::shared_ptr<const void> > >::pointer _M_end_of_storage
// CHECK-NEXT: 136 |   class std::vector<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> > MArgs
// CHECK-NEXT: 136 |     struct std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> > (base)
// CHECK-NEXT: 136 |       struct std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> >::_Vector_impl _M_impl
// CHECK-NEXT: 136 |         class std::allocator<class cl::sycl::detail::ArgDesc> (base) (empty)
// CHECK-NEXT: 136 |           class __gnu_cxx::new_allocator<class cl::sycl::detail::ArgDesc> (base) (empty)
// CHECK-NEXT: 136 |         std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> >::pointer _M_start
// CHECK-NEXT: 144 |         std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> >::pointer _M_finish
// CHECK-NEXT: 152 |         std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> >::pointer _M_end_of_storage
// CHECK-NEXT: 160 |   class std::vector<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> > MAssociatedAccesors
// CHECK-NEXT: 160 |     struct std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> > (base)
// CHECK-NEXT: 160 |       struct std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> >::_Vector_impl _M_impl
// CHECK-NEXT: 160 |         class std::allocator<class cl::sycl::detail::ArgDesc> (base) (empty)
// CHECK-NEXT: 160 |           class __gnu_cxx::new_allocator<class cl::sycl::detail::ArgDesc> (base) (empty)
// CHECK-NEXT: 160 |         std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> >::pointer _M_start
// CHECK-NEXT: 168 |         std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> >::pointer _M_finish
// CHECK-NEXT: 176 |         std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> >::pointer _M_end_of_storage
// CHECK-NEXT: 184 |   class std::vector<class cl::sycl::detail::AccessorImplHost *, class std::allocator<class cl::sycl::detail::AccessorImplHost *> > MRequirements
// CHECK-NEXT: 184 |     struct std::_Vector_base<class cl::sycl::detail::AccessorImplHost *, class std::allocator<class cl::sycl::detail::AccessorImplHost *> > (base)
// CHECK-NEXT: 184 |       struct std::_Vector_base<class cl::sycl::detail::AccessorImplHost *, class std::allocator<class cl::sycl::detail::AccessorImplHost *> >::_Vector_impl _M_impl
// CHECK-NEXT: 184 |         class std::allocator<class cl::sycl::detail::AccessorImplHost *> (base) (empty)
// CHECK-NEXT: 184 |           class __gnu_cxx::new_allocator<class cl::sycl::detail::AccessorImplHost *> (base) (empty)
// CHECK-NEXT: 184 |         std::_Vector_base<class cl::sycl::detail::AccessorImplHost *, class std::allocator<class cl::sycl::detail::AccessorImplHost *> >::pointer _M_start
// CHECK-NEXT: 192 |         std::_Vector_base<class cl::sycl::detail::AccessorImplHost *, class std::allocator<class cl::sycl::detail::AccessorImplHost *> >::pointer _M_finish
// CHECK-NEXT: 200 |         std::_Vector_base<class cl::sycl::detail::AccessorImplHost *, class std::allocator<class cl::sycl::detail::AccessorImplHost *> >::pointer _M_end_of_storage
// CHECK-NEXT: 208 |   class cl::sycl::detail::NDRDescT MNDRDesc
// CHECK-NEXT: 208 |     class cl::sycl::range<3> GlobalSize
// CHECK-NEXT: 208 |       class cl::sycl::detail::array<3> (base)
// CHECK-NEXT: 208 |         size_t [3] common_array
// CHECK-NEXT: 232 |     class cl::sycl::range<3> LocalSize
// CHECK-NEXT: 232 |       class cl::sycl::detail::array<3> (base)
// CHECK-NEXT: 232 |         size_t [3] common_array
// CHECK-NEXT: 256 |     class cl::sycl::id<3> GlobalOffset
// CHECK-NEXT: 256 |       class cl::sycl::detail::array<3> (base)
// CHECK-NEXT: 256 |         size_t [3] common_array
// CHECK-NEXT: 280 |     class cl::sycl::range<3> NumWorkGroups
// CHECK-NEXT: 280 |       class cl::sycl::detail::array<3> (base)
// CHECK-NEXT: 280 |         size_t [3] common_array
// CHECK-NEXT: 304 |     size_t Dims
// CHECK-NEXT: 312 |   class std::__cxx11::basic_string<char, struct std::char_traits<char>, class std::allocator<char> > MKernelName
// CHECK-NEXT: 312 |     struct std::__cxx11::basic_string<char, struct std::char_traits<char>, class std::allocator<char> >::_Alloc_hider _M_dataplus
// CHECK-NEXT: 312 |       class std::allocator<char> (base) (empty)
// CHECK-NEXT: 312 |         class __gnu_cxx::new_allocator<char> (base) (empty)
// CHECK-NEXT: 312 |       std::__cxx11::basic_string<char, struct std::char_traits<char>, class std::allocator<char> >::pointer _M_p
// CHECK-NEXT: 320 |     std::__cxx11::basic_string<char, struct std::char_traits<char>, class std::allocator<char> >::size_type _M_string_length
// CHECK-NEXT: 328 |     union std::__cxx11::basic_string<char, struct std::char_traits<char>, class std::allocator<char> >::
// CHECK-NEXT: 328 |       char [16] _M_local_buf
// CHECK-NEXT: 328 |       std::__cxx11::basic_string<char, struct std::char_traits<char>, class std::allocator<char> >::size_type _M_allocated_capacity
// CHECK-NEXT: 344 |   class std::shared_ptr<class cl::sycl::detail::kernel_impl> MKernel
// CHECK-NEXT: 344 |     class std::__shared_ptr<class cl::sycl::detail::kernel_impl, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT: 344 |       class std::__shared_ptr_access<class cl::sycl::detail::kernel_impl, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT: 344 |       std::__shared_ptr<class cl::sycl::detail::kernel_impl, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT: 352 |       class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 352 |         _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 360 |   detail::class CG::CGTYPE MCGType
// CHECK-NEXT: 368 |   void * MSrcPtr
// CHECK-NEXT: 376 |   void * MDstPtr
// CHECK-NEXT: 384 |   size_t MLength
// CHECK-NEXT: 392 |   class std::vector<char, class std::allocator<char> > MPattern
// CHECK-NEXT: 392 |     struct std::_Vector_base<char, class std::allocator<char> > (base)
// CHECK-NEXT: 392 |       struct std::_Vector_base<char, class std::allocator<char> >::_Vector_impl _M_impl
// CHECK-NEXT: 392 |         class std::allocator<char> (base) (empty)
// CHECK-NEXT: 392 |           class __gnu_cxx::new_allocator<char> (base) (empty)
// CHECK-NEXT: 392 |         std::_Vector_base<char, class std::allocator<char> >::pointer _M_start
// CHECK-NEXT: 400 |         std::_Vector_base<char, class std::allocator<char> >::pointer _M_finish
// CHECK-NEXT: 408 |         std::_Vector_base<char, class std::allocator<char> >::pointer _M_end_of_storage
// CHECK-NEXT: 416 |   class std::unique_ptr<class cl::sycl::detail::HostKernelBase, struct std::default_delete<class cl::sycl::detail::HostKernelBase> > MHostKernel
// CHECK-NEXT: 416 |     class std::__uniq_ptr_impl<class cl::sycl::detail::HostKernelBase, struct std::default_delete<class cl::sycl::detail::HostKernelBase> > _M_t
// CHECK-NEXT: 416 |       class std::tuple<class cl::sycl::detail::HostKernelBase *, struct std::default_delete<class cl::sycl::detail::HostKernelBase> > _M_t
// CHECK-NEXT: 416 |         struct std::_Tuple_impl<0, class cl::sycl::detail::HostKernelBase *, struct std::default_delete<class cl::sycl::detail::HostKernelBase> > (base)
// CHECK-NEXT: 416 |           struct std::_Tuple_impl<1, struct std::default_delete<class cl::sycl::detail::HostKernelBase> > (base) (empty)
// CHECK-NEXT: 416 |             struct std::_Head_base<1, struct std::default_delete<class cl::sycl::detail::HostKernelBase>, true> (base) (empty)
// CHECK-NEXT: 416 |               struct std::default_delete<class cl::sycl::detail::HostKernelBase> (base) (empty)
// CHECK-NEXT: 416 |           struct std::_Head_base<0, class cl::sycl::detail::HostKernelBase *, false> (base)
// CHECK-NEXT: 416 |             class cl::sycl::detail::HostKernelBase * _M_head_impl
// CHECK-NEXT: 424 |   class std::unique_ptr<class cl::sycl::detail::HostTask, struct std::default_delete<class cl::sycl::detail::HostTask> > MHostTask
// CHECK-NEXT: 424 |     class std::__uniq_ptr_impl<class cl::sycl::detail::HostTask, struct std::default_delete<class cl::sycl::detail::HostTask> > _M_t
// CHECK-NEXT: 424 |       class std::tuple<class cl::sycl::detail::HostTask *, struct std::default_delete<class cl::sycl::detail::HostTask> > _M_t
// CHECK-NEXT: 424 |         struct std::_Tuple_impl<0, class cl::sycl::detail::HostTask *, struct std::default_delete<class cl::sycl::detail::HostTask> > (base)
// CHECK-NEXT: 424 |           struct std::_Tuple_impl<1, struct std::default_delete<class cl::sycl::detail::HostTask> > (base) (empty)
// CHECK-NEXT: 424 |             struct std::_Head_base<1, struct std::default_delete<class cl::sycl::detail::HostTask>, true> (base) (empty)
// CHECK-NEXT: 424 |               struct std::default_delete<class cl::sycl::detail::HostTask> (base) (empty)
// CHECK-NEXT: 424 |           struct std::_Head_base<0, class cl::sycl::detail::HostTask *, false> (base)
// CHECK-NEXT: 424 |             class cl::sycl::detail::HostTask * _M_head_impl
// CHECK-NEXT: 432 |   detail::OSModuleHandle MOSModuleHandle
// CHECK-NEXT: 440 |   class std::unique_ptr<class cl::sycl::detail::InteropTask, struct std::default_delete<class cl::sycl::detail::InteropTask> > MInteropTask
// CHECK-NEXT: 440 |     class std::__uniq_ptr_impl<class cl::sycl::detail::InteropTask, struct std::default_delete<class cl::sycl::detail::InteropTask> > _M_t
// CHECK-NEXT: 440 |       class std::tuple<class cl::sycl::detail::InteropTask *, struct std::default_delete<class cl::sycl::detail::InteropTask> > _M_t
// CHECK-NEXT: 440 |         struct std::_Tuple_impl<0, class cl::sycl::detail::InteropTask *, struct std::default_delete<class cl::sycl::detail::InteropTask> > (base)
// CHECK-NEXT: 440 |           struct std::_Tuple_impl<1, struct std::default_delete<class cl::sycl::detail::InteropTask> > (base) (empty)
// CHECK-NEXT: 440 |             struct std::_Head_base<1, struct std::default_delete<class cl::sycl::detail::InteropTask>, true> (base) (empty)
// CHECK-NEXT: 440 |               struct std::default_delete<class cl::sycl::detail::InteropTask> (base) (empty)
// CHECK-NEXT: 440 |           struct std::_Head_base<0, class cl::sycl::detail::InteropTask *, false> (base)
// CHECK-NEXT: 440 |             class cl::sycl::detail::InteropTask * _M_head_impl
// CHECK-NEXT: 448 |   class std::vector<class std::shared_ptr<class cl::sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::event_impl> > > MEvents
// CHECK-NEXT: 448 |     struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::event_impl> > > (base)
// CHECK-NEXT: 448 |       struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::event_impl> > >::_Vector_impl _M_impl
// CHECK-NEXT: 448 |         class std::allocator<class std::shared_ptr<class cl::sycl::detail::event_impl> > (base) (empty)
// CHECK-NEXT: 448 |           class __gnu_cxx::new_allocator<class std::shared_ptr<class cl::sycl::detail::event_impl> > (base) (empty)
// CHECK-NEXT: 448 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::event_impl> > >::pointer _M_start
// CHECK-NEXT: 456 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::event_impl> > >::pointer _M_finish
// CHECK-NEXT: 464 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::event_impl> > >::pointer _M_end_of_storage
// CHECK-NEXT: 472 |   _Bool MIsHost
// CHECK-NEXT: 480 |   struct cl::sycl::detail::code_location MCodeLoc
// CHECK-NEXT: 480 |     const char * MFileName
// CHECK-NEXT: 488 |     const char * MFunctionName
// CHECK-NEXT: 496 |     unsigned long MLineNo
// CHECK-NEXT: 504 |     unsigned long MColumnNo
// CHECK-NEXT: 512 |   _Bool MIsFinalized
// CHECK-NEXT: 520 |   class cl::sycl::event MLastEvent
// CHECK-NEXT: 520 |     class std::shared_ptr<class cl::sycl::detail::event_impl> impl
// CHECK-NEXT: 520 |       class std::__shared_ptr<class cl::sycl::detail::event_impl, __gnu_cxx::_S_atomic> (base)
// CHECK-NEXT: 520 |         class std::__shared_ptr_access<class cl::sycl::detail::event_impl, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CHECK-NEXT: 520 |         std::__shared_ptr<class cl::sycl::detail::event_impl, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CHECK-NEXT: 528 |         class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CHECK-NEXT: 528 |           _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT:     | [sizeof=536, dsize=536, align=8,
// CHECK-NEXT:     |  nvsize=536, nvalign=8]
