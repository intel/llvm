// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

#include <sycl/handler.hpp>
#include <sycl/queue.hpp>

void foo() {
  sycl::queue Q;
  Q.submit([](sycl::handler &CGH) { CGH.single_task<class Test>([]() {}); });
}

// clang-format off

// The order of field declarations and their types are important.
// CHECK:        0 | class sycl::handler
// CHECK-NEXT:   0 |   class std::shared_ptr<class sycl::detail::handler_impl> impl
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
// CHECK-NEXT:  32 |   class std::vector<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > MLocalAccStorage
// CHECK-NEXT:  32 |     struct std::_Vector_base<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > > (base)
// CHECK-NEXT:  32 |       struct std::_Vector_base<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > >::_Vector_impl _M_impl
// CHECK-NEXT:  32 |         class std::allocator<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > (base) (empty)
// CHECK:       32 |           pointer _M_start
// CHECK-NEXT:  40 |           pointer _M_finish
// CHECK-NEXT:  48 |           pointer _M_end_of_storage
// CHECK-NEXT:  56 |   class std::vector<class std::shared_ptr<class sycl::detail::stream_impl> > MStreamStorage
// CHECK-NEXT:  56 |     struct std::_Vector_base<class std::shared_ptr<class sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::stream_impl> > > (base)
// CHECK-NEXT:  56 |       struct std::_Vector_base<class std::shared_ptr<class sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::stream_impl> > >::_Vector_impl _M_impl
// CHECK-NEXT:  56 |         class std::allocator<class std::shared_ptr<class sycl::detail::stream_impl> > (base) (empty)
// CHECK:       56 |           pointer _M_start
// CHECK-NEXT:  64 |           pointer _M_finish
// CHECK-NEXT:  72 |           pointer _M_end_of_storage
// CHECK-NEXT:  80 |   class sycl::detail::string MKernelName
// CHECK-NEXT:  80 |     char * str
// CHECK-NEXT:  88 |   class std::shared_ptr<class sycl::detail::kernel_impl> MKernel
// CHECK-NEXT:  88 |     class std::__shared_ptr<class sycl::detail::kernel_impl> (base)
// CHECK-NEXT:  88 |       class std::__shared_ptr_access<class sycl::detail::kernel_impl, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT:  88 |       element_type * _M_ptr
// CHECK-NEXT:  96 |       class std::__shared_count<> _M_refcount
// CHECK-NEXT:  96 |         _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT: 104 |   void * MSrcPtr
// CHECK-NEXT: 112 |   void * MDstPtr
// CHECK-NEXT: 120 |   size_t MLength
// CHECK-NEXT: 128 |   class std::vector<unsigned char> MPattern
// CHECK-NEXT: 128 |     struct std::_Vector_base<unsigned char, class std::allocator<unsigned char> > (base)
// CHECK-NEXT: 128 |       struct std::_Vector_base<unsigned char, class std::allocator<unsigned char> >::_Vector_impl _M_impl
// CHECK-NEXT: 128 |         class std::allocator<unsigned char> (base) (empty)
// CHECK:      128 |           pointer _M_start
// CHECK-NEXT: 136 |           pointer _M_finish
// CHECK-NEXT: 144 |           pointer _M_end_of_storage
// CHECK-NEXT: 152 |   class std::unique_ptr<class sycl::detail::HostKernelBase> MHostKernel
// CHECK:      152 |       class std::__uniq_ptr_impl<class sycl::detail::HostKernelBase, struct std::default_delete<class sycl::detail::HostKernelBase> >
// CHECK-NEXT: 152 |         class std::tuple<class sycl::detail::HostKernelBase *, struct std::default_delete<class sycl::detail::HostKernelBase> > _M_t
// CHECK-NEXT: 152 |           struct std::_Tuple_impl<0, class sycl::detail::HostKernelBase *, struct std::default_delete<class sycl::detail::HostKernelBase> > (base)
// CHECK-NEXT: 152 |             struct std::_Tuple_impl<1, struct std::default_delete<class sycl::detail::HostKernelBase> > (base) (empty)
// CHECK:      152 |             struct std::_Head_base<0, class sycl::detail::HostKernelBase *> (base)
// CHECK-NEXT: 152 |               class sycl::detail::HostKernelBase * _M_head_impl
// CHECK-NEXT: 160 |   struct sycl::detail::code_location MCodeLoc
// CHECK-NEXT: 160 |     const char * MFileName
// CHECK-NEXT: 168 |     const char * MFunctionName
// CHECK-NEXT: 176 |     unsigned long MLineNo
// CHECK-NEXT: 184 |     unsigned long MColumnNo
// CHECK-NEXT: 192 |   _Bool MIsFinalized
// CHECK-NEXT: 200 |   class sycl::event MLastEvent
// CHECK-NEXT: 200 |     class sycl::detail::OwnerLessBase<class sycl::event> (base) (empty)
// CHECK-NEXT: 200 |     class std::shared_ptr<class sycl::detail::event_impl> impl
// CHECK-NEXT: 200 |       class std::__shared_ptr<class sycl::detail::event_impl> (base)
// CHECK-NEXT: 200 |         class std::__shared_ptr_access<class sycl::detail::event_impl, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT: 200 |         element_type * _M_ptr
// CHECK-NEXT: 208 |         class std::__shared_count<> _M_refcount
// CHECK-NEXT: 208 |           _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT:     | [sizeof=216, dsize=216, align=8,
// CHECK-NEXT:     |  nvsize=216, nvalign=8]
