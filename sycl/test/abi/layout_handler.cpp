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
// CHECK-NEXT:   0 |   class std::unique_ptr<class sycl::detail::handler_impl> implOwner
// CHECK-NEXT:   0 |     struct std::__uniq_ptr_data<class sycl::detail::handler_impl, struct std::default_delete<class sycl::detail::handler_impl> > _M_t
// CHECK-NEXT:   0 |       class std::__uniq_ptr_impl<class sycl::detail::handler_impl, struct std::default_delete<class sycl::detail::handler_impl> > (base)
// CHECK-NEXT:   0 |         class std::tuple<class sycl::detail::handler_impl *, struct std::default_delete<class sycl::detail::handler_impl> > _M_t
// CHECK-NEXT:   0 |           struct std::_Tuple_impl<0, class sycl::detail::handler_impl *, struct std::default_delete<class sycl::detail::handler_impl> > (base)
// CHECK-NEXT:   0 |             struct std::_Tuple_impl<1, struct std::default_delete<class sycl::detail::handler_impl> > (base) (empty)
// CHECK:        0 |               struct std::_Head_base<1, struct std::default_delete<class sycl::detail::handler_impl> > (base) (empty)
// CHECK-NEXT:   0 |                 struct std::default_delete<class sycl::detail::handler_impl> _M_head_impl (empty)
// CHECK:        0 |             struct std::_Head_base<0, class sycl::detail::handler_impl *> (base)
// CHECK-NEXT:   0 |               class sycl::detail::handler_impl * _M_head_impl
// CHECK-NEXT:   8 |   detail::handler_impl * impl
// CHECK-NEXT:   16 |   class std::vector<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > MLocalAccStorage
// CHECK-NEXT:   16 |     struct std::_Vector_base<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > > (base)
// CHECK-NEXT:   16 |       struct std::_Vector_base<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > >::_Vector_impl _M_impl
// CHECK-NEXT:   16 |         class std::allocator<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > (base) (empty)
// CHECK-NEXT:   16 |           class std::__new_allocator<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > (base) (empty)
// CHECK-NEXT:   16 |         struct std::_Vector_base<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > >::_Vector_impl_data (base)
// CHECK:        16 |           pointer _M_start
// CHECK-NEXT:   24 |           pointer _M_finish
// CHECK-NEXT:   32 |           pointer _M_end_of_storage
// CHECK-NEXT:   40 |   class std::vector<class std::shared_ptr<class sycl::detail::stream_impl> > MStreamStorage
// CHECK-NEXT:   40 |     struct std::_Vector_base<class std::shared_ptr<class sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::stream_impl> > > (base)
// CHECK-NEXT:   40 |       struct std::_Vector_base<class std::shared_ptr<class sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::stream_impl> > >::_Vector_impl _M_impl
// CHECK-NEXT:   40 |         class std::allocator<class std::shared_ptr<class sycl::detail::stream_impl> > (base) (empty)
// CHECK-NEXT:   40 |           class std::__new_allocator<class std::shared_ptr<class sycl::detail::stream_impl> > (base) (empty)
// CHECK-NEXT:   40 |         struct std::_Vector_base<class std::shared_ptr<class sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::stream_impl> > >::_Vector_impl_data (base)
// CHECK:        40 |           pointer _M_start
// CHECK-NEXT:   48 |           pointer _M_finish
// CHECK-NEXT:   56 |           pointer _M_end_of_storage
// CHECK-NEXT:   64 |   class sycl::detail::string MKernelName
// CHECK-NEXT:   64 |     char * str
// CHECK-NEXT:   72 |   class std::shared_ptr<class sycl::detail::kernel_impl> MKernel
// CHECK-NEXT:   72 |     class std::__shared_ptr<class sycl::detail::kernel_impl> (base)
// CHECK-NEXT:   72 |       class std::__shared_ptr_access<class sycl::detail::kernel_impl, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT:   72 |       element_type * _M_ptr
// CHECK-NEXT:   80 |       class std::__shared_count<> _M_refcount
// CHECK-NEXT:   80 |         _Sp_counted_base<(enum __gnu_cxx::_Lock_policy)2U> * _M_pi
// CHECK-NEXT:   88 |   void * MSrcPtr
// CHECK-NEXT:   96 |   void * MDstPtr
// CHECK-NEXT:   104 |   size_t MLength
// CHECK-NEXT:   112 |   class std::vector<unsigned char> MPattern
// CHECK-NEXT:   112 |     struct std::_Vector_base<unsigned char, class std::allocator<unsigned char> > (base)
// CHECK-NEXT:   112 |       struct std::_Vector_base<unsigned char, class std::allocator<unsigned char> >::_Vector_impl _M_impl
// CHECK-NEXT:   112 |         class std::allocator<unsigned char> (base) (empty)
// CHECK-NEXT:   112 |           class std::__new_allocator<unsigned char> (base) (empty)
// CHECK-NEXT:   112 |         struct std::_Vector_base<unsigned char, class std::allocator<unsigned char> >::_Vector_impl_data (base)
// CHECK:        112 |           pointer _M_start
// CHECK-NEXT:   120 |           pointer _M_finish
// CHECK-NEXT:   128 |           pointer _M_end_of_storage
// CHECK-NEXT:   136 |   class std::unique_ptr<class sycl::detail::HostKernelBase> MHostKernel
// CHECK-NEXT:   136 |     struct std::__uniq_ptr_data<class sycl::detail::HostKernelBase, struct std::default_delete<class sycl::detail::HostKernelBase> > _M_t
// CHECK:        136 |       class std::__uniq_ptr_impl<class sycl::detail::HostKernelBase, struct std::default_delete<class sycl::detail::HostKernelBase> > (base)
// CHECK-NEXT:   136 |         class std::tuple<class sycl::detail::HostKernelBase *, struct std::default_delete<class sycl::detail::HostKernelBase> > _M_t
// CHECK-NEXT:   136 |           struct std::_Tuple_impl<0, class sycl::detail::HostKernelBase *, struct std::default_delete<class sycl::detail::HostKernelBase> > (base)
// CHECK-NEXT:   136 |             struct std::_Tuple_impl<1, struct std::default_delete<class sycl::detail::HostKernelBase> > (base) (empty)
// CHECK:        136 |               struct std::_Head_base<1, struct std::default_delete<class sycl::detail::HostKernelBase> > (base) (empty)
// CHECK-NEXT:   136 |                 struct std::default_delete<class sycl::detail::HostKernelBase> _M_head_impl (empty)
// CHECK:        136 |             struct std::_Head_base<0, class sycl::detail::HostKernelBase *> (base)
// CHECK-NEXT:   136 |               class sycl::detail::HostKernelBase * _M_head_impl
// CHECK-NEXT:   144 |   struct sycl::detail::code_location MCodeLoc
// CHECK-NEXT:   144 |     const char * MFileName
// CHECK-NEXT:   152 |     const char * MFunctionName
// CHECK-NEXT:   160 |     unsigned long MLineNo
// CHECK-NEXT:   168 |     unsigned long MColumnNo
// CHECK-NEXT:     | [sizeof=176, dsize=176, align=8,
// CHECK-NEXT:     |  nvsize=176, nvalign=8]
// clang-format on
