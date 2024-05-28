// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

#include <sycl/handler.hpp>
#include <sycl/queue.hpp>

void foo() {
  sycl::queue Q;
  Q.submit([](sycl::handler &CGH) {
    CGH.single_task<class Test>([]() {});
  });
}

// clang-format off

// The order of field declarations and their types are important.
// CHECK:        0 | class sycl::handler
// CHECK-NEXT:   0 |   class std::shared_ptr<class sycl::detail::handler_impl> MImpl
// CHECK-NEXT:   0 |     class std::__shared_ptr<class sycl::detail::handler_impl> (base)
// CHECK-NEXT:   0 |       class std::__shared_ptr_access<class sycl::detail::handler_impl, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT:   0 |       element_type * _M_ptr
// CHECK-NEXT:   8 |       class std::__shared_count<> _M_refcount
// CHECK-NEXT:   8 |         _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT:  16 |   class std::shared_ptr<class sycl::detail::queue_impl> MQueue
// CHECK-NEXT:  16 |     class std::__shared_ptr<class sycl::detail::queue_impl> (base)
// CHECK-NEXT:  16 |       class std::__shared_ptr_access<class sycl::detail::queue_impl, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT:  16 |       element_type * _M_ptr
// CHECK-NEXT:  24 |       class std::__shared_count<> _M_refcount
// CHECK-NEXT:  24 |         _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT:  32 |   struct sycl::detail::CG::StorageInitHelper CGData
// CHECK-NEXT:  32 |     class std::vector<class std::vector<char> > MArgsStorage
// CHECK-NEXT:  32 |       struct std::_Vector_base<class std::vector<char>, class std::allocator<class std::vector<char> > > (base)
// CHECK-NEXT:  32 |         struct std::_Vector_base<class std::vector<char>, class std::allocator<class std::vector<char> > >::_Vector_impl _M_impl
// CHECK-NEXT:  32 |           class std::allocator<class std::vector<char> > (base) (empty)
// CHECK:       32 |             pointer _M_start
// CHECK-NEXT:  40 |             pointer _M_finish
// CHECK-NEXT:  48 |             pointer _M_end_of_storage
// CHECK-NEXT:  56 |     class std::vector<class std::shared_ptr<class sycl::detail::AccessorImplHost> > MAccStorage
// CHECK-NEXT:  56 |       struct std::_Vector_base<class std::shared_ptr<class sycl::detail::AccessorImplHost>, class std::allocator<class std::shared_ptr<class sycl::detail::AccessorImplHost> > > (base)
// CHECK-NEXT:  56 |         struct std::_Vector_base<class std::shared_ptr<class sycl::detail::AccessorImplHost>, class std::allocator<class std::shared_ptr<class sycl::detail::AccessorImplHost> > >::_Vector_impl _M_impl
// CHECK-NEXT:  56 |           class std::allocator<class std::shared_ptr<class sycl::detail::AccessorImplHost> > (base) (empty)
// CHECK:       56 |             pointer _M_start
// CHECK-NEXT:  64 |             pointer _M_finish
// CHECK-NEXT:  72 |             pointer _M_end_of_storage
// CHECK-NEXT:  80 |     class std::vector<class std::shared_ptr<const void> > MSharedPtrStorage
// CHECK-NEXT:  80 |       struct std::_Vector_base<class std::shared_ptr<const void>, class std::allocator<class std::shared_ptr<const void> > > (base)
// CHECK-NEXT:  80 |         struct std::_Vector_base<class std::shared_ptr<const void>, class std::allocator<class std::shared_ptr<const void> > >::_Vector_impl _M_impl
// CHECK-NEXT:  80 |           class std::allocator<class std::shared_ptr<const void> > (base) (empty)
// CHECK:       80 |             pointer _M_start
// CHECK-NEXT:  88 |             pointer _M_finish
// CHECK-NEXT:  96 |             pointer _M_end_of_storage
// CHECK-NEXT: 104 |     class std::vector<class sycl::detail::AccessorImplHost *> MRequirements
// CHECK-NEXT: 104 |       struct std::_Vector_base<class sycl::detail::AccessorImplHost *, class std::allocator<class sycl::detail::AccessorImplHost *> > (base)
// CHECK-NEXT: 104 |         struct std::_Vector_base<class sycl::detail::AccessorImplHost *, class std::allocator<class sycl::detail::AccessorImplHost *> >::_Vector_impl _M_impl
// CHECK-NEXT: 104 |           class std::allocator<class sycl::detail::AccessorImplHost *> (base) (empty)
// CHECK:      104 |             pointer _M_start
// CHECK-NEXT: 112 |             pointer _M_finish
// CHECK-NEXT: 120 |             pointer _M_end_of_storage
// CHECK-NEXT: 128 |     class std::vector<class std::shared_ptr<class sycl::detail::event_impl> > MEvents
// CHECK-NEXT: 128 |       struct std::_Vector_base<class std::shared_ptr<class sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::event_impl> > > (base)
// CHECK-NEXT: 128 |         struct std::_Vector_base<class std::shared_ptr<class sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::event_impl> > >::_Vector_impl _M_impl
// CHECK-NEXT: 128 |           class std::allocator<class std::shared_ptr<class sycl::detail::event_impl> > (base) (empty)
// CHECK:      128 |             pointer _M_start
// CHECK-NEXT: 136 |             pointer _M_finish
// CHECK-NEXT: 144 |             pointer _M_end_of_storage
// CHECK-NEXT: 152 |   class std::vector<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > MLocalAccStorage
// CHECK-NEXT: 152 |     struct std::_Vector_base<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > > (base)
// CHECK-NEXT: 152 |       struct std::_Vector_base<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > >::_Vector_impl _M_impl
// CHECK-NEXT: 152 |         class std::allocator<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > (base) (empty)
// CHECK:      152 |           pointer _M_start
// CHECK-NEXT: 160 |           pointer _M_finish
// CHECK-NEXT: 168 |           pointer _M_end_of_storage
// CHECK-NEXT: 176 |   class std::vector<class std::shared_ptr<class sycl::detail::stream_impl> > MStreamStorage
// CHECK-NEXT: 176 |     struct std::_Vector_base<class std::shared_ptr<class sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::stream_impl> > > (base)
// CHECK-NEXT: 176 |       struct std::_Vector_base<class std::shared_ptr<class sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::stream_impl> > >::_Vector_impl _M_impl
// CHECK-NEXT: 176 |         class std::allocator<class std::shared_ptr<class sycl::detail::stream_impl> > (base) (empty)
// CHECK:      176 |           pointer _M_start
// CHECK-NEXT: 184 |           pointer _M_finish
// CHECK-NEXT: 192 |           pointer _M_end_of_storage
// CHECK-NEXT: 200 |   class std::vector<class sycl::detail::ArgDesc> MArgs
// CHECK-NEXT: 200 |     struct std::_Vector_base<class sycl::detail::ArgDesc, class std::allocator<class sycl::detail::ArgDesc> > (base)
// CHECK-NEXT: 200 |       struct std::_Vector_base<class sycl::detail::ArgDesc, class std::allocator<class sycl::detail::ArgDesc> >::_Vector_impl _M_impl
// CHECK-NEXT: 200 |         class std::allocator<class sycl::detail::ArgDesc> (base) (empty)
// CHECK:      200 |           pointer _M_start
// CHECK-NEXT: 208 |           pointer _M_finish
// CHECK-NEXT: 216 |           pointer _M_end_of_storage
// CHECK-NEXT: 224 |   class std::vector<class sycl::detail::ArgDesc> MAssociatedAccesors
// CHECK-NEXT: 224 |     struct std::_Vector_base<class sycl::detail::ArgDesc, class std::allocator<class sycl::detail::ArgDesc> > (base)
// CHECK-NEXT: 224 |       struct std::_Vector_base<class sycl::detail::ArgDesc, class std::allocator<class sycl::detail::ArgDesc> >::_Vector_impl _M_impl
// CHECK-NEXT: 224 |         class std::allocator<class sycl::detail::ArgDesc> (base) (empty)
// CHECK:      224 |           pointer _M_start
// CHECK-NEXT: 232 |           pointer _M_finish
// CHECK-NEXT: 240 |           pointer _M_end_of_storage
// CHECK-NEXT: 248 |   class sycl::detail::NDRDescT MNDRDesc
// CHECK-NEXT: 248 |     class sycl::range<3> GlobalSize
// CHECK-NEXT: 248 |       class sycl::detail::array<3> (base)
// CHECK-NEXT: 248 |         size_t[3] common_array
// CHECK-NEXT: 272 |     class sycl::range<3> LocalSize
// CHECK-NEXT: 272 |       class sycl::detail::array<3> (base)
// CHECK-NEXT: 272 |         size_t[3] common_array
// CHECK-NEXT: 296 |     class sycl::id<3> GlobalOffset
// CHECK-NEXT: 296 |       class sycl::detail::array<3> (base)
// CHECK-NEXT: 296 |         size_t[3] common_array
// CHECK-NEXT: 320 |     class sycl::range<3> NumWorkGroups
// CHECK-NEXT: 320 |       class sycl::detail::array<3> (base)
// CHECK-NEXT: 320 |         size_t[3] common_array
// CHECK-NEXT: 344 |     size_t Dims
// CHECK-NEXT: 352 |   class sycl::detail::string MKernelName
// CHECK-NEXT: 352 |     char * str
// CHECK-NEXT: 360 |   class std::shared_ptr<class sycl::detail::kernel_impl> MKernel
// CHECK-NEXT: 360 |     class std::__shared_ptr<class sycl::detail::kernel_impl> (base)
// CHECK-NEXT: 360 |       class std::__shared_ptr_access<class sycl::detail::kernel_impl, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT: 360 |       element_type * _M_ptr
// CHECK-NEXT: 368 |       class std::__shared_count<> _M_refcount
// CHECK-NEXT: 368 |         _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 376 |   detail::class CG::CGTYPE MCGType
// CHECK-NEXT: 384 |   void * MSrcPtr
// CHECK-NEXT: 392 |   void * MDstPtr
// CHECK-NEXT: 400 |   size_t MLength
// CHECK-NEXT: 408 |   class std::vector<char> MPattern
// CHECK-NEXT: 408 |     struct std::_Vector_base<char, class std::allocator<char> > (base)
// CHECK-NEXT: 408 |       struct std::_Vector_base<char, class std::allocator<char> >::_Vector_impl _M_impl
// CHECK-NEXT: 408 |         class std::allocator<char> (base) (empty)
// CHECK:      408 |           pointer _M_start
// CHECK-NEXT: 416 |           pointer _M_finish
// CHECK-NEXT: 424 |           pointer _M_end_of_storage
// CHECK-NEXT: 432 |   class std::unique_ptr<class sycl::detail::HostKernelBase> MHostKernel
// CHECK:      432 |       class std::__uniq_ptr_impl<class sycl::detail::HostKernelBase, struct std::default_delete<class sycl::detail::HostKernelBase> >
// CHECK-NEXT: 432 |         class std::tuple<class sycl::detail::HostKernelBase *, struct std::default_delete<class sycl::detail::HostKernelBase> > _M_t
// CHECK-NEXT: 432 |           struct std::_Tuple_impl<0, class sycl::detail::HostKernelBase *, struct std::default_delete<class sycl::detail::HostKernelBase> > (base)
// CHECK-NEXT: 432 |             struct std::_Tuple_impl<1, struct std::default_delete<class sycl::detail::HostKernelBase> > (base) (empty)
// CHECK:      432 |             struct std::_Head_base<0, class sycl::detail::HostKernelBase *> (base)
// CHECK-NEXT: 432 |               class sycl::detail::HostKernelBase * _M_head_impl
// CHECK-NEXT: 440 |   class std::unique_ptr<class sycl::detail::HostTask> MHostTask
// CHECK:      440 |       class std::__uniq_ptr_impl<class sycl::detail::HostTask, struct std::default_delete<class sycl::detail::HostTask> >
// CHECK-NEXT: 440 |         class std::tuple<class sycl::detail::HostTask *, struct std::default_delete<class sycl::detail::HostTask> > _M_t
// CHECK-NEXT: 440 |           struct std::_Tuple_impl<0, class sycl::detail::HostTask *, struct std::default_delete<class sycl::detail::HostTask> > (base)
// CHECK-NEXT: 440 |             struct std::_Tuple_impl<1, struct std::default_delete<class sycl::detail::HostTask> > (base) (empty)
// CHECK:      440 |             struct std::_Head_base<0, class sycl::detail::HostTask *> (base)
// CHECK-NEXT: 440 |               class sycl::detail::HostTask * _M_head_impl
// CHECK-NEXT: 448 |   class std::vector<class std::shared_ptr<class sycl::detail::event_impl> > MEventsWaitWithBarrier
// CHECK-NEXT: 448 |     struct std::_Vector_base<class std::shared_ptr<class sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::event_impl> > > (base)
// CHECK-NEXT: 448 |       struct std::_Vector_base<class std::shared_ptr<class sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::event_impl> > >::_Vector_impl _M_impl
// CHECK-NEXT: 448 |         class std::allocator<class std::shared_ptr<class sycl::detail::event_impl> > (base) (empty)
// CHECK:      448 |           pointer _M_start
// CHECK-NEXT: 456 |           pointer _M_finish
// CHECK-NEXT: 464 |           pointer _M_end_of_storage
// CHECK-NEXT: 472 |   class std::shared_ptr<class sycl::ext::oneapi::experimental::detail::graph_impl> MGraph
// CHECK-NEXT: 472 |     class std::__shared_ptr<class sycl::ext::oneapi::experimental::detail::graph_impl> (base)
// CHECK-NEXT: 472 |       class std::__shared_ptr_access<class sycl::ext::oneapi::experimental::detail::graph_impl, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT: 472 |       element_type * _M_ptr
// CHECK-NEXT: 480 |       class std::__shared_count<> _M_refcount
// CHECK-NEXT: 480 |         _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 488 |   class std::shared_ptr<class sycl::ext::oneapi::experimental::detail::exec_graph_impl> MExecGraph
// CHECK-NEXT: 488 |     class std::__shared_ptr<class sycl::ext::oneapi::experimental::detail::exec_graph_impl> (base)
// CHECK-NEXT: 488 |       class std::__shared_ptr_access<class sycl::ext::oneapi::experimental::detail::exec_graph_impl, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT: 488 |       element_type * _M_ptr
// CHECK-NEXT: 496 |       class std::__shared_count<> _M_refcount
// CHECK-NEXT: 496 |         _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 504 |   class std::shared_ptr<class sycl::ext::oneapi::experimental::detail::node_impl> MSubgraphNode
// CHECK-NEXT: 504 |     class std::__shared_ptr<class sycl::ext::oneapi::experimental::detail::node_impl> (base)
// CHECK-NEXT: 504 |       class std::__shared_ptr_access<class sycl::ext::oneapi::experimental::detail::node_impl, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT: 504 |       element_type * _M_ptr
// CHECK-NEXT: 512 |       class std::__shared_count<> _M_refcount
// CHECK-NEXT: 512 |         _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 520 |   class std::unique_ptr<class sycl::detail::CG> MGraphNodeCG
// CHECK:      520 |       class std::__uniq_ptr_impl<class sycl::detail::CG, struct std::default_delete<class sycl::detail::CG> >
// CHECK-NEXT: 520 |         class std::tuple<class sycl::detail::CG *, struct std::default_delete<class sycl::detail::CG> > _M_t
// CHECK-NEXT: 520 |           struct std::_Tuple_impl<0, class sycl::detail::CG *, struct std::default_delete<class sycl::detail::CG> > (base)
// CHECK-NEXT: 520 |             struct std::_Tuple_impl<1, struct std::default_delete<class sycl::detail::CG> > (base) (empty)
// CHECK:      520 |             struct std::_Head_base<0, class sycl::detail::CG *> (base)
// CHECK-NEXT: 520 |               class sycl::detail::CG * _M_head_impl
// CHECK-NEXT: 528 |   _Bool MIsHost
// CHECK-NEXT: 536 |   struct sycl::detail::code_location MCodeLoc
// CHECK-NEXT: 536 |     const char * MFileName
// CHECK-NEXT: 544 |     const char * MFunctionName
// CHECK-NEXT: 552 |     unsigned long MLineNo
// CHECK-NEXT: 560 |     unsigned long MColumnNo
// CHECK-NEXT: 568 |   _Bool MIsFinalized
// CHECK-NEXT: 576 |   class sycl::event MLastEvent
// CHECK-NEXT: 576 |     class sycl::detail::OwnerLessBase<class sycl::event> (base) (empty)
// CHECK-NEXT: 576 |     class std::shared_ptr<class sycl::detail::event_impl> impl
// CHECK-NEXT: 576 |       class std::__shared_ptr<class sycl::detail::event_impl> (base)
// CHECK-NEXT: 576 |         class std::__shared_ptr_access<class sycl::detail::event_impl, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT: 576 |         element_type * _M_ptr
// CHECK-NEXT: 584 |         class std::__shared_count<> _M_refcount
// CHECK-NEXT: 584 |           _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT:     | [sizeof=592, dsize=592, align=8,
// CHECK-NEXT:     |  nvsize=592, nvalign=8]
