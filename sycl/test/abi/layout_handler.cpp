// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s | FileCheck %s
// REQUIRES: linux

#include <CL/sycl/handler.hpp>
#include <CL/sycl/queue.hpp>

void foo() {
  sycl::queue Q;
  Q.submit([](sycl::handler &CGH) {
    CGH.single_task<class Test>([](){});
  });
}

// The order of field declarations and their types are important.

// CHECK: 0 | class cl::sycl::handler
// CEHCK-NEXT: 0 |   class std::shared_ptr<class cl::sycl::detail::queue_impl> MQueue
// CEHCK-NEXT: 0 |     class std::__shared_ptr<class cl::sycl::detail::queue_impl, __gnu_cxx::_S_atomic> (base)
// CEHCK-NEXT: 0 |       class std::__shared_ptr_access<class cl::sycl::detail::queue_impl, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CEHCK-NEXT: 0 |       std::__shared_ptr<class cl::sycl::detail::queue_impl, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CEHCK-NEXT: 8 |       class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CEHCK-NEXT: 8 |         _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CEHCK-NEXT: 16 |   class std::vector<class std::vector<char, class std::allocator<char> >, class std::allocator<class std::vector<char, class std::allocator<char> > > > MArgsStorage
// CEHCK-NEXT: 16 |     struct std::_Vector_base<class std::vector<char, class std::allocator<char> >, class std::allocator<class std::vector<char, class std::allocator<char> > > > (base)
// CEHCK-NEXT: 16 |       struct std::_Vector_base<class std::vector<char, class std::allocator<char> >, class std::allocator<class std::vector<char, class std::allocator<char> > > >::_Vector_impl _M_impl
// CEHCK-NEXT: 16 |         class std::allocator<class std::vector<char, class std::allocator<char> > > (base) (empty)
// CEHCK-NEXT: 16 |           class __gnu_cxx::new_allocator<class std::vector<char, class std::allocator<char> > > (base) (empty)
// CEHCK-NEXT: 16 |         std::_Vector_base<class std::vector<char, class std::allocator<char> >, class std::allocator<class std::vector<char, class std::allocator<char> > > >::pointer _M_start
// CEHCK-NEXT: 24 |         std::_Vector_base<class std::vector<char, class std::allocator<char> >, class std::allocator<class std::vector<char, class std::allocator<char> > > >::pointer _M_finish
// CEHCK-NEXT: 32 |         std::_Vector_base<class std::vector<char, class std::allocator<char> >, class std::allocator<class std::vector<char, class std::allocator<char> > > >::pointer _M_end_of_storage
// CEHCK-NEXT: 40 |   class std::vector<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> > > MAccStorage
// CEHCK-NEXT: 40 |     struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> > > (base)
// CEHCK-NEXT: 40 |       struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> > >::_Vector_impl _M_impl
// CEHCK-NEXT: 40 |         class std::allocator<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> > (base) (empty)
// CEHCK-NEXT: 40 |           class __gnu_cxx::new_allocator<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> > (base) (empty)
// CEHCK-NEXT: 40 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> > >::pointer _M_start
// CEHCK-NEXT: 48 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> > >::pointer _M_finish
// CEHCK-NEXT: 56 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::AccessorImplHost> > >::pointer _M_end_of_storage
// CEHCK-NEXT: 64 |   class std::vector<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost> > > MLocalAccStorage
// CEHCK-NEXT: 64 |     struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost> > > (base)
// CEHCK-NEXT: 64 |       struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost> > >::_Vector_impl _M_impl
// CEHCK-NEXT: 64 |         class std::allocator<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost> > (base) (empty)
// CEHCK-NEXT: 64 |           class __gnu_cxx::new_allocator<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost> > (base) (empty)
// CEHCK-NEXT: 64 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost> > >::pointer _M_start
// CEHCK-NEXT: 72 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost> > >::pointer _M_finish
// CEHCK-NEXT: 80 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::LocalAccessorImplHost> > >::pointer _M_end_of_storage
// CEHCK-NEXT: 88 |   class std::vector<class std::shared_ptr<class cl::sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::stream_impl> > > MStreamStorage
// CEHCK-NEXT: 88 |     struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::stream_impl> > > (base)
// CEHCK-NEXT: 88 |       struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::stream_impl> > >::_Vector_impl _M_impl
// CEHCK-NEXT: 88 |         class std::allocator<class std::shared_ptr<class cl::sycl::detail::stream_impl> > (base) (empty)
// CEHCK-NEXT: 88 |           class __gnu_cxx::new_allocator<class std::shared_ptr<class cl::sycl::detail::stream_impl> > (base) (empty)
// CEHCK-NEXT: 88 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::stream_impl> > >::pointer _M_start
// CEHCK-NEXT: 96 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::stream_impl> > >::pointer _M_finish
// CEHCK-NEXT: 104 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::stream_impl> > >::pointer _M_end_of_storage
// CEHCK-NEXT: 112 |   class std::vector<class std::shared_ptr<const void>, class std::allocator<class std::shared_ptr<const void> > > MSharedPtrStorage
// CEHCK-NEXT: 112 |     struct std::_Vector_base<class std::shared_ptr<const void>, class std::allocator<class std::shared_ptr<const void> > > (base)
// CEHCK-NEXT: 112 |       struct std::_Vector_base<class std::shared_ptr<const void>, class std::allocator<class std::shared_ptr<const void> > >::_Vector_impl _M_impl
// CEHCK-NEXT: 112 |         class std::allocator<class std::shared_ptr<const void> > (base) (empty)
// CEHCK-NEXT: 112 |           class __gnu_cxx::new_allocator<class std::shared_ptr<const void> > (base) (empty)
// CEHCK-NEXT: 112 |         std::_Vector_base<class std::shared_ptr<const void>, class std::allocator<class std::shared_ptr<const void> > >::pointer _M_start
// CEHCK-NEXT: 120 |         std::_Vector_base<class std::shared_ptr<const void>, class std::allocator<class std::shared_ptr<const void> > >::pointer _M_finish
// CEHCK-NEXT: 128 |         std::_Vector_base<class std::shared_ptr<const void>, class std::allocator<class std::shared_ptr<const void> > >::pointer _M_end_of_storage
// CEHCK-NEXT: 136 |   class std::vector<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> > MArgs
// CEHCK-NEXT: 136 |     struct std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> > (base)
// CEHCK-NEXT: 136 |       struct std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> >::_Vector_impl _M_impl
// CEHCK-NEXT: 136 |         class std::allocator<class cl::sycl::detail::ArgDesc> (base) (empty)
// CEHCK-NEXT: 136 |           class __gnu_cxx::new_allocator<class cl::sycl::detail::ArgDesc> (base) (empty)
// CEHCK-NEXT: 136 |         std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> >::pointer _M_start
// CEHCK-NEXT: 144 |         std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> >::pointer _M_finish
// CEHCK-NEXT: 152 |         std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> >::pointer _M_end_of_storage
// CEHCK-NEXT: 160 |   class std::vector<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> > MAssociatedAccesors
// CEHCK-NEXT: 160 |     struct std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> > (base)
// CEHCK-NEXT: 160 |       struct std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> >::_Vector_impl _M_impl
// CEHCK-NEXT: 160 |         class std::allocator<class cl::sycl::detail::ArgDesc> (base) (empty)
// CEHCK-NEXT: 160 |           class __gnu_cxx::new_allocator<class cl::sycl::detail::ArgDesc> (base) (empty)
// CEHCK-NEXT: 160 |         std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> >::pointer _M_start
// CEHCK-NEXT: 168 |         std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> >::pointer _M_finish
// CEHCK-NEXT: 176 |         std::_Vector_base<class cl::sycl::detail::ArgDesc, class std::allocator<class cl::sycl::detail::ArgDesc> >::pointer _M_end_of_storage
// CEHCK-NEXT: 184 |   class std::vector<class cl::sycl::detail::AccessorImplHost *, class std::allocator<class cl::sycl::detail::AccessorImplHost *> > MRequirements
// CEHCK-NEXT: 184 |     struct std::_Vector_base<class cl::sycl::detail::AccessorImplHost *, class std::allocator<class cl::sycl::detail::AccessorImplHost *> > (base)
// CEHCK-NEXT: 184 |       struct std::_Vector_base<class cl::sycl::detail::AccessorImplHost *, class std::allocator<class cl::sycl::detail::AccessorImplHost *> >::_Vector_impl _M_impl
// CEHCK-NEXT: 184 |         class std::allocator<class cl::sycl::detail::AccessorImplHost *> (base) (empty)
// CEHCK-NEXT: 184 |           class __gnu_cxx::new_allocator<class cl::sycl::detail::AccessorImplHost *> (base) (empty)
// CEHCK-NEXT: 184 |         std::_Vector_base<class cl::sycl::detail::AccessorImplHost *, class std::allocator<class cl::sycl::detail::AccessorImplHost *> >::pointer _M_start
// CEHCK-NEXT: 192 |         std::_Vector_base<class cl::sycl::detail::AccessorImplHost *, class std::allocator<class cl::sycl::detail::AccessorImplHost *> >::pointer _M_finish
// CEHCK-NEXT: 200 |         std::_Vector_base<class cl::sycl::detail::AccessorImplHost *, class std::allocator<class cl::sycl::detail::AccessorImplHost *> >::pointer _M_end_of_storage
// CEHCK-NEXT: 208 |   class cl::sycl::detail::NDRDescT MNDRDesc
// CEHCK-NEXT: 208 |     class cl::sycl::range<3> GlobalSize
// CEHCK-NEXT: 208 |       class cl::sycl::detail::array<3> (base)
// CEHCK-NEXT: 208 |         size_t [3] common_array
// CEHCK-NEXT: 232 |     class cl::sycl::range<3> LocalSize
// CEHCK-NEXT: 232 |       class cl::sycl::detail::array<3> (base)
// CEHCK-NEXT: 232 |         size_t [3] common_array
// CEHCK-NEXT: 256 |     class cl::sycl::id<3> GlobalOffset
// CEHCK-NEXT: 256 |       class cl::sycl::detail::array<3> (base)
// CEHCK-NEXT: 256 |         size_t [3] common_array
// CEHCK-NEXT: 280 |     class cl::sycl::range<3> NumWorkGroups
// CEHCK-NEXT: 280 |       class cl::sycl::detail::array<3> (base)
// CEHCK-NEXT: 280 |         size_t [3] common_array
// CEHCK-NEXT: 304 |     size_t Dims
// CEHCK-NEXT: 312 |   class std::__cxx11::basic_string<char, struct std::char_traits<char>, class std::allocator<char> > MKernelName
// CEHCK-NEXT: 312 |     struct std::__cxx11::basic_string<char, struct std::char_traits<char>, class std::allocator<char> >::_Alloc_hider _M_dataplus
// CEHCK-NEXT: 312 |       class std::allocator<char> (base) (empty)
// CEHCK-NEXT: 312 |         class __gnu_cxx::new_allocator<char> (base) (empty)
// CEHCK-NEXT: 312 |       std::__cxx11::basic_string<char, struct std::char_traits<char>, class std::allocator<char> >::pointer _M_p
// CEHCK-NEXT: 320 |     std::__cxx11::basic_string<char, struct std::char_traits<char>, class std::allocator<char> >::size_type _M_string_length
// CEHCK-NEXT: 328 |     union std::__cxx11::basic_string<char, struct std::char_traits<char>, class std::allocator<char> >::(anonymous at /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/bits/basic_string.h:160:7)
// CEHCK-NEXT: 328 |       char [16] _M_local_buf
// CEHCK-NEXT: 328 |       std::__cxx11::basic_string<char, struct std::char_traits<char>, class std::allocator<char> >::size_type _M_allocated_capacity
// CEHCK-NEXT: 344 |   class std::shared_ptr<class cl::sycl::detail::kernel_impl> MKernel
// CEHCK-NEXT: 344 |     class std::__shared_ptr<class cl::sycl::detail::kernel_impl, __gnu_cxx::_S_atomic> (base)
// CEHCK-NEXT: 344 |       class std::__shared_ptr_access<class cl::sycl::detail::kernel_impl, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CEHCK-NEXT: 344 |       std::__shared_ptr<class cl::sycl::detail::kernel_impl, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CEHCK-NEXT: 352 |       class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CEHCK-NEXT: 352 |         _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CEHCK-NEXT: 360 |   detail::class CG::CGTYPE MCGType
// CEHCK-NEXT: 368 |   void * MSrcPtr
// CEHCK-NEXT: 376 |   void * MDstPtr
// CEHCK-NEXT: 384 |   size_t MLength
// CEHCK-NEXT: 392 |   class std::vector<char, class std::allocator<char> > MPattern
// CEHCK-NEXT: 392 |     struct std::_Vector_base<char, class std::allocator<char> > (base)
// CEHCK-NEXT: 392 |       struct std::_Vector_base<char, class std::allocator<char> >::_Vector_impl _M_impl
// CEHCK-NEXT: 392 |         class std::allocator<char> (base) (empty)
// CEHCK-NEXT: 392 |           class __gnu_cxx::new_allocator<char> (base) (empty)
// CEHCK-NEXT: 392 |         std::_Vector_base<char, class std::allocator<char> >::pointer _M_start
// CEHCK-NEXT: 400 |         std::_Vector_base<char, class std::allocator<char> >::pointer _M_finish
// CEHCK-NEXT: 408 |         std::_Vector_base<char, class std::allocator<char> >::pointer _M_end_of_storage
// CEHCK-NEXT: 416 |   class std::unique_ptr<class cl::sycl::detail::HostKernelBase, struct std::default_delete<class cl::sycl::detail::HostKernelBase> > MHostKernel
// CEHCK-NEXT: 416 |     class std::__uniq_ptr_impl<class cl::sycl::detail::HostKernelBase, struct std::default_delete<class cl::sycl::detail::HostKernelBase> > _M_t
// CEHCK-NEXT: 416 |       class std::tuple<class cl::sycl::detail::HostKernelBase *, struct std::default_delete<class cl::sycl::detail::HostKernelBase> > _M_t
// CEHCK-NEXT: 416 |         struct std::_Tuple_impl<0, class cl::sycl::detail::HostKernelBase *, struct std::default_delete<class cl::sycl::detail::HostKernelBase> > (base)
// CEHCK-NEXT: 416 |           struct std::_Tuple_impl<1, struct std::default_delete<class cl::sycl::detail::HostKernelBase> > (base) (empty)
// CEHCK-NEXT: 416 |             struct std::_Head_base<1, struct std::default_delete<class cl::sycl::detail::HostKernelBase>, true> (base) (empty)
// CEHCK-NEXT: 416 |               struct std::default_delete<class cl::sycl::detail::HostKernelBase> (base) (empty)
// CEHCK-NEXT: 416 |           struct std::_Head_base<0, class cl::sycl::detail::HostKernelBase *, false> (base)
// CEHCK-NEXT: 416 |             class cl::sycl::detail::HostKernelBase * _M_head_impl
// CEHCK-NEXT: 424 |   class std::unique_ptr<class cl::sycl::detail::HostTask, struct std::default_delete<class cl::sycl::detail::HostTask> > MHostTask
// CEHCK-NEXT: 424 |     class std::__uniq_ptr_impl<class cl::sycl::detail::HostTask, struct std::default_delete<class cl::sycl::detail::HostTask> > _M_t
// CEHCK-NEXT: 424 |       class std::tuple<class cl::sycl::detail::HostTask *, struct std::default_delete<class cl::sycl::detail::HostTask> > _M_t
// CEHCK-NEXT: 424 |         struct std::_Tuple_impl<0, class cl::sycl::detail::HostTask *, struct std::default_delete<class cl::sycl::detail::HostTask> > (base)
// CEHCK-NEXT: 424 |           struct std::_Tuple_impl<1, struct std::default_delete<class cl::sycl::detail::HostTask> > (base) (empty)
// CEHCK-NEXT: 424 |             struct std::_Head_base<1, struct std::default_delete<class cl::sycl::detail::HostTask>, true> (base) (empty)
// CEHCK-NEXT: 424 |               struct std::default_delete<class cl::sycl::detail::HostTask> (base) (empty)
// CEHCK-NEXT: 424 |           struct std::_Head_base<0, class cl::sycl::detail::HostTask *, false> (base)
// CEHCK-NEXT: 424 |             class cl::sycl::detail::HostTask * _M_head_impl
// CEHCK-NEXT: 432 |   detail::OSModuleHandle MOSModuleHandle
// CEHCK-NEXT: 440 |   class std::unique_ptr<class cl::sycl::detail::InteropTask, struct std::default_delete<class cl::sycl::detail::InteropTask> > MInteropTask
// CEHCK-NEXT: 440 |     class std::__uniq_ptr_impl<class cl::sycl::detail::InteropTask, struct std::default_delete<class cl::sycl::detail::InteropTask> > _M_t
// CEHCK-NEXT: 440 |       class std::tuple<class cl::sycl::detail::InteropTask *, struct std::default_delete<class cl::sycl::detail::InteropTask> > _M_t
// CEHCK-NEXT: 440 |         struct std::_Tuple_impl<0, class cl::sycl::detail::InteropTask *, struct std::default_delete<class cl::sycl::detail::InteropTask> > (base)
// CEHCK-NEXT: 440 |           struct std::_Tuple_impl<1, struct std::default_delete<class cl::sycl::detail::InteropTask> > (base) (empty)
// CEHCK-NEXT: 440 |             struct std::_Head_base<1, struct std::default_delete<class cl::sycl::detail::InteropTask>, true> (base) (empty)
// CEHCK-NEXT: 440 |               struct std::default_delete<class cl::sycl::detail::InteropTask> (base) (empty)
// CEHCK-NEXT: 440 |           struct std::_Head_base<0, class cl::sycl::detail::InteropTask *, false> (base)
// CEHCK-NEXT: 440 |             class cl::sycl::detail::InteropTask * _M_head_impl
// CEHCK-NEXT: 448 |   class std::vector<class std::shared_ptr<class cl::sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::event_impl> > > MEvents
// CEHCK-NEXT: 448 |     struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::event_impl> > > (base)
// CEHCK-NEXT: 448 |       struct std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::event_impl> > >::_Vector_impl _M_impl
// CEHCK-NEXT: 448 |         class std::allocator<class std::shared_ptr<class cl::sycl::detail::event_impl> > (base) (empty)
// CEHCK-NEXT: 448 |           class __gnu_cxx::new_allocator<class std::shared_ptr<class cl::sycl::detail::event_impl> > (base) (empty)
// CEHCK-NEXT: 448 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::event_impl> > >::pointer _M_start
// CEHCK-NEXT: 456 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::event_impl> > >::pointer _M_finish
// CEHCK-NEXT: 464 |         std::_Vector_base<class std::shared_ptr<class cl::sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class cl::sycl::detail::event_impl> > >::pointer _M_end_of_storage
// CEHCK-NEXT: 472 |   _Bool MIsHost
// CEHCK-NEXT: 480 |   struct cl::sycl::detail::code_location MCodeLoc
// CEHCK-NEXT: 480 |     const char * MFileName
// CEHCK-NEXT: 488 |     const char * MFunctionName
// CEHCK-NEXT: 496 |     unsigned long MLineNo
// CEHCK-NEXT: 504 |     unsigned long MColumnNo
// CEHCK-NEXT: 512 |   _Bool MIsFinalized
// CEHCK-NEXT: 520 |   class cl::sycl::event MLastEvent
// CEHCK-NEXT: 520 |     class std::shared_ptr<class cl::sycl::detail::event_impl> impl
// CEHCK-NEXT: 520 |       class std::__shared_ptr<class cl::sycl::detail::event_impl, __gnu_cxx::_S_atomic> (base)
// CEHCK-NEXT: 520 |         class std::__shared_ptr_access<class cl::sycl::detail::event_impl, __gnu_cxx::_S_atomic, false, false> (base) (empty)
// CEHCK-NEXT: 520 |         std::__shared_ptr<class cl::sycl::detail::event_impl, __gnu_cxx::_S_atomic>::element_type * _M_ptr
// CEHCK-NEXT: 528 |         class std::__shared_count<__gnu_cxx::_S_atomic> _M_refcount
// CEHCK-NEXT: 528 |           _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CEHCK-NEXT:     | [sizeof=536, dsize=536, align=8,
// CEHCK-NEXT:     |  nvsize=536, nvalign=8]
