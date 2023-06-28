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
// CHECK-NEXT:  32 |             class __gnu_cxx::new_allocator<class std::vector<char> > (base) (empty)
// CHECK:       32 |             pointer _M_start
// CHECK-NEXT:  40 |             pointer _M_finish
// CHECK-NEXT:  48 |             pointer _M_end_of_storage
// CHECK-NEXT:  56 |     class std::vector<class std::shared_ptr<class sycl::detail::AccessorImplHost> > MAccStorage
// CHECK-NEXT:  56 |       struct std::_Vector_base<class std::shared_ptr<class sycl::detail::AccessorImplHost>, class std::allocator<class std::shared_ptr<class sycl::detail::AccessorImplHost> > > (base)
// CHECK-NEXT:  56 |         struct std::_Vector_base<class std::shared_ptr<class sycl::detail::AccessorImplHost>, class std::allocator<class std::shared_ptr<class sycl::detail::AccessorImplHost> > >::_Vector_impl _M_impl
// CHECK-NEXT:  56 |           class std::allocator<class std::shared_ptr<class sycl::detail::AccessorImplHost> > (base) (empty)
// CHECK-NEXT:  56 |             class __gnu_cxx::new_allocator<class std::shared_ptr<class sycl::detail::AccessorImplHost> > (base) (empty)
// CHECK:       56 |             pointer _M_start
// CHECK-NEXT:  64 |             pointer _M_finish
// CHECK-NEXT:  72 |             pointer _M_end_of_storage
// CHECK-NEXT:  80 |     class std::vector<class std::shared_ptr<const void> > MSharedPtrStorage
// CHECK-NEXT:  80 |       struct std::_Vector_base<class std::shared_ptr<const void>, class std::allocator<class std::shared_ptr<const void> > > (base)
// CHECK-NEXT:  80 |         struct std::_Vector_base<class std::shared_ptr<const void>, class std::allocator<class std::shared_ptr<const void> > >::_Vector_impl _M_impl
// CHECK-NEXT:  80 |           class std::allocator<class std::shared_ptr<const void> > (base) (empty)
// CHECK-NEXT:  80 |             class __gnu_cxx::new_allocator<class std::shared_ptr<const void> > (base) (empty)
// CHECK:       80 |             pointer _M_start
// CHECK-NEXT:  88 |             pointer _M_finish
// CHECK-NEXT:  96 |             pointer _M_end_of_storage
// CHECK-NEXT:       104 |     class std::unordered_set<class sycl::detail::AccessorImplHost *> MRequirements
// CHECK-NEXT:       104 |       class std::_Hashtable<class sycl::detail::AccessorImplHost *, class sycl::detail::AccessorImplHost *, class std::allocator<class sycl::detail::AccessorImplHost *>, struct std::__detail::_Identity, struct std::equal_to<class sycl::detail::AccessorImplHost *>, struct std::hash<class sycl::detail::AccessorImplHost *>, struct std::__detail::_Mod_range_hashing, struct std::__detail::_Default_ranged_hash, struct std::__detail::_Prime_rehash_policy, struct std::__detail::_Hashtable_traits<false, true, true> > _M_h
// CHECK-NEXT:       104 |         struct std::__detail::_Hashtable_base<class sycl::detail::AccessorImplHost *, class sycl::detail::AccessorImplHost *, struct std::__detail::_Identity, struct std::equal_to<class sycl::detail::AccessorImplHost *>, struct std::hash<class sycl::detail::AccessorImplHost *>, struct std::__detail::_Mod_range_hashing, struct std::__detail::_Default_ranged_hash, struct std::__detail::_Hashtable_traits<false, true, true> > (base) (empty)
// CHECK-NEXT:       104 |           struct std::__detail::_Hash_code_base<class sycl::detail::AccessorImplHost *, class sycl::detail::AccessorImplHost *, struct std::__detail::_Identity, struct std::hash<class sycl::detail::AccessorImplHost *>, struct std::__detail::_Mod_range_hashing, struct std::__detail::_Default_ranged_hash, false> (base) (empty)
// CHECK-NEXT:       104 |             struct std::__detail::_Hashtable_ebo_helper<1, struct std::hash<class sycl::detail::AccessorImplHost *> > (base) (empty)
// CHECK-NEXT:       104 |               struct std::hash<class sycl::detail::AccessorImplHost *> (base) (empty)
// CHECK-NEXT:       104 |                 struct std::__hash_base<unsigned long, class sycl::detail::AccessorImplHost *> (base) (empty)
// CHECK-NEXT:       104 |           struct std::__detail::_Hashtable_ebo_helper<0, struct std::equal_to<class sycl::detail::AccessorImplHost *> > (base) (empty)
// CHECK-NEXT:       104 |             struct std::equal_to<class sycl::detail::AccessorImplHost *> (base) (empty)
// CHECK-NEXT:       104 |               struct std::binary_function<class sycl::detail::AccessorImplHost *, class sycl::detail::AccessorImplHost *, _Bool> (base) (empty)
// CHECK-NEXT:       104 |         struct std::__detail::_Map_base<class sycl::detail::AccessorImplHost *, class sycl::detail::AccessorImplHost *, class std::allocator<class sycl::detail::AccessorImplHost *>, struct std::__detail::_Identity, struct std::equal_to<class sycl::detail::AccessorImplHost *>, struct std::hash<class sycl::detail::AccessorImplHost *>, struct std::__detail::_Mod_range_hashing, struct std::__detail::_Default_ranged_hash, struct std::__detail::_Prime_rehash_policy, struct std::__detail::_Hashtable_traits<false, true, true> > (base) (empty)
// CHECK-NEXT:       104 |         struct std::__detail::_Insert<class sycl::detail::AccessorImplHost *, class sycl::detail::AccessorImplHost *, class std::allocator<class sycl::detail::AccessorImplHost *>, struct std::__detail::_Identity, struct std::equal_to<class sycl::detail::AccessorImplHost *>, struct std::hash<class sycl::detail::AccessorImplHost *>, struct std::__detail::_Mod_range_hashing, struct std::__detail::_Default_ranged_hash, struct std::__detail::_Prime_rehash_policy, struct std::__detail::_Hashtable_traits<false, true, true> > (base) (empty)
// CHECK-NEXT:       104 |           struct std::__detail::_Insert_base<class sycl::detail::AccessorImplHost *, class sycl::detail::AccessorImplHost *, class std::allocator<class sycl::detail::AccessorImplHost *>, struct std::__detail::_Identity, struct std::equal_to<class sycl::detail::AccessorImplHost *>, struct std::hash<class sycl::detail::AccessorImplHost *>, struct std::__detail::_Mod_range_hashing, struct std::__detail::_Default_ranged_hash, struct std::__detail::_Prime_rehash_policy, struct std::__detail::_Hashtable_traits<false, true, true> > (base) (empty)
// CHECK-NEXT:       104 |         struct std::__detail::_Rehash_base<class sycl::detail::AccessorImplHost *, class sycl::detail::AccessorImplHost *, class std::allocator<class sycl::detail::AccessorImplHost *>, struct std::__detail::_Identity, struct std::equal_to<class sycl::detail::AccessorImplHost *>, struct std::hash<class sycl::detail::AccessorImplHost *>, struct std::__detail::_Mod_range_hashing, struct std::__detail::_Default_ranged_hash, struct std::__detail::_Prime_rehash_policy, struct std::__detail::_Hashtable_traits<false, true, true> > (base) (empty)
// CHECK-NEXT:       104 |         struct std::__detail::_Equality<class sycl::detail::AccessorImplHost *, class sycl::detail::AccessorImplHost *, class std::allocator<class sycl::detail::AccessorImplHost *>, struct std::__detail::_Identity, struct std::equal_to<class sycl::detail::AccessorImplHost *>, struct std::hash<class sycl::detail::AccessorImplHost *>, struct std::__detail::_Mod_range_hashing, struct std::__detail::_Default_ranged_hash, struct std::__detail::_Prime_rehash_policy, struct std::__detail::_Hashtable_traits<false, true, true> > (base) (empty)
// CHECK-NEXT:       104 |         struct std::__detail::_Hashtable_alloc<class std::allocator<struct std::__detail::_Hash_node<class sycl::detail::AccessorImplHost *, false> > > (base) (empty)
// CHECK-NEXT:       104 |           struct std::__detail::_Hashtable_ebo_helper<0, class std::allocator<struct std::__detail::_Hash_node<class sycl::detail::AccessorImplHost *, false> > > (base) (empty)
// CHECK-NEXT:       104 |             class std::allocator<struct std::__detail::_Hash_node<class sycl::detail::AccessorImplHost *, false> > (base) (empty)
// CHECK-NEXT:       104 |               class __gnu_cxx::new_allocator<struct std::__detail::_Hash_node<class sycl::detail::AccessorImplHost *, false> > (base) (empty)
// CHECK-NEXT:       104 |         struct std::_Enable_default_constructor<true, struct std::__detail::_Hash_node_base> (base) (empty)
// CHECK-NEXT:       104 |         __buckets_ptr _M_buckets
// CHECK-NEXT:       112 |         size_type _M_bucket_count
// CHECK-NEXT:       120 |         struct std::__detail::_Hash_node_base _M_before_begin
// CHECK-NEXT:       120 |           _Hash_node_base * _M_nxt
// CHECK-NEXT:       128 |         size_type _M_element_count
// CHECK-NEXT:       136 |         struct std::__detail::_Prime_rehash_policy _M_rehash_policy
// CHECK-NEXT:       136 |           float _M_max_load_factor
// CHECK-NEXT:       144 |           std::size_t _M_next_resize
// CHECK-NEXT:       152 |         __node_base_ptr _M_single_bucket
// CHECK-NEXT:       160 |     class std::vector<class std::shared_ptr<class sycl::detail::event_impl> > MEvents
// CHECK-NEXT:       160 |       struct std::_Vector_base<class std::shared_ptr<class sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::event_impl> > > (base)
// CHECK-NEXT:       160 |         struct std::_Vector_base<class std::shared_ptr<class sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::event_impl> > >::_Vector_impl _M_impl
// CHECK-NEXT:       160 |           class std::allocator<class std::shared_ptr<class sycl::detail::event_impl> > (base) (empty)
// CHECK-NEXT:       160 |             class __gnu_cxx::new_allocator<class std::shared_ptr<class sycl::detail::event_impl> > (base) (empty)
// CHECK-NEXT:       160 |           struct std::_Vector_base<class std::shared_ptr<class sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::event_impl> > >::_Vector_impl_data (base)
// CHECK:       160 |             pointer _M_start
// CHECK-NEXT:       168 |             pointer _M_finish
// CHECK-NEXT:       176 |             pointer _M_end_of_storage
// CHECK-NEXT:       184 |   class std::vector<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > MLocalAccStorage
// CHECK-NEXT:       184 |     struct std::_Vector_base<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > > (base)
// CHECK-NEXT:       184 |       struct std::_Vector_base<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > >::_Vector_impl _M_impl
// CHECK-NEXT:       184 |         class std::allocator<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > (base) (empty)
// CHECK-NEXT:       184 |           class __gnu_cxx::new_allocator<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > (base) (empty)
// CHECK-NEXT:       184 |         struct std::_Vector_base<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost>, class std::allocator<class std::shared_ptr<class sycl::detail::LocalAccessorImplHost> > >::_Vector_impl_data (base)
// CHECK:       184 |           pointer _M_start
// CHECK-NEXT:       192 |           pointer _M_finish
// CHECK-NEXT:       200 |           pointer _M_end_of_storage
// CHECK-NEXT:       208 |   class std::vector<class std::shared_ptr<class sycl::detail::stream_impl> > MStreamStorage
// CHECK-NEXT:       208 |     struct std::_Vector_base<class std::shared_ptr<class sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::stream_impl> > > (base)
// CHECK-NEXT:       208 |       struct std::_Vector_base<class std::shared_ptr<class sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::stream_impl> > >::_Vector_impl _M_impl
// CHECK-NEXT:       208 |         class std::allocator<class std::shared_ptr<class sycl::detail::stream_impl> > (base) (empty)
// CHECK-NEXT:       208 |           class __gnu_cxx::new_allocator<class std::shared_ptr<class sycl::detail::stream_impl> > (base) (empty)
// CHECK-NEXT:       208 |         struct std::_Vector_base<class std::shared_ptr<class sycl::detail::stream_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::stream_impl> > >::_Vector_impl_data (base)
// CHECK:       208 |           pointer _M_start
// CHECK-NEXT:       216 |           pointer _M_finish
// CHECK-NEXT:       224 |           pointer _M_end_of_storage
// CHECK-NEXT:       232 |   class std::vector<class sycl::detail::ArgDesc> MArgs
// CHECK-NEXT:       232 |     struct std::_Vector_base<class sycl::detail::ArgDesc, class std::allocator<class sycl::detail::ArgDesc> > (base)
// CHECK-NEXT:       232 |       struct std::_Vector_base<class sycl::detail::ArgDesc, class std::allocator<class sycl::detail::ArgDesc> >::_Vector_impl _M_impl
// CHECK-NEXT:       232 |         class std::allocator<class sycl::detail::ArgDesc> (base) (empty)
// CHECK-NEXT:       232 |           class __gnu_cxx::new_allocator<class sycl::detail::ArgDesc> (base) (empty)
// CHECK-NEXT:       232 |         struct std::_Vector_base<class sycl::detail::ArgDesc, class std::allocator<class sycl::detail::ArgDesc> >::_Vector_impl_data (base)
// CHECK:       232 |           pointer _M_start
// CHECK-NEXT:       240 |           pointer _M_finish
// CHECK-NEXT:       248 |           pointer _M_end_of_storage
// CHECK-NEXT:       256 |   class std::vector<class sycl::detail::ArgDesc> MAssociatedAccesors
// CHECK-NEXT:       256 |     struct std::_Vector_base<class sycl::detail::ArgDesc, class std::allocator<class sycl::detail::ArgDesc> > (base)
// CHECK-NEXT:       256 |       struct std::_Vector_base<class sycl::detail::ArgDesc, class std::allocator<class sycl::detail::ArgDesc> >::_Vector_impl _M_impl
// CHECK-NEXT:       256 |         class std::allocator<class sycl::detail::ArgDesc> (base) (empty)
// CHECK-NEXT:       256 |           class __gnu_cxx::new_allocator<class sycl::detail::ArgDesc> (base) (empty)
// CHECK-NEXT:       256 |         struct std::_Vector_base<class sycl::detail::ArgDesc, class std::allocator<class sycl::detail::ArgDesc> >::_Vector_impl_data (base)
// CHECK:       256 |           pointer _M_start
// CHECK-NEXT:       264 |           pointer _M_finish
// CHECK-NEXT:       272 |           pointer _M_end_of_storage
// CHECK-NEXT:       280 |   class sycl::detail::NDRDescT MNDRDesc
// CHECK-NEXT:       280 |     class sycl::range<3> GlobalSize
// CHECK-NEXT:       280 |       class sycl::detail::array<3> (base)
// CHECK-NEXT:       280 |         size_t[3] common_array
// CHECK-NEXT:       304 |     class sycl::range<3> LocalSize
// CHECK-NEXT:       304 |       class sycl::detail::array<3> (base)
// CHECK-NEXT:       304 |         size_t[3] common_array
// CHECK-NEXT:       328 |     class sycl::id<3> GlobalOffset
// CHECK-NEXT:       328 |       class sycl::detail::array<3> (base)
// CHECK-NEXT:       328 |         size_t[3] common_array
// CHECK-NEXT:       352 |     class sycl::range<3> NumWorkGroups
// CHECK-NEXT:       352 |       class sycl::detail::array<3> (base)
// CHECK-NEXT:       352 |         size_t[3] common_array
// CHECK-NEXT:       376 |     size_t Dims
// CHECK-NEXT:       384 |   class std::basic_string<char> MKernelName
// CHECK-NEXT:       384 |     struct std::basic_string<char>::_Alloc_hider _M_dataplus
// CHECK-NEXT:       384 |       class std::allocator<char> (base) (empty)
// CHECK-NEXT:       384 |         class __gnu_cxx::new_allocator<char> (base) (empty)
// CHECK-NEXT:       384 |       pointer _M_p
// CHECK-NEXT:       392 |     size_type _M_string_length
// CHECK-NEXT:       400 |     union std::basic_string<char>::(anonymous at /usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/bits/basic_string.h:179:7)
// CHECK-NEXT:       400 |       char[16] _M_local_buf
// CHECK-NEXT:       400 |       size_type _M_allocated_capacity
// CHECK-NEXT:       416 |   class std::shared_ptr<class sycl::detail::kernel_impl> MKernel
// CHECK-NEXT:       416 |     class std::__shared_ptr<class sycl::detail::kernel_impl> (base)
// CHECK-NEXT:       416 |       class std::__shared_ptr_access<class sycl::detail::kernel_impl, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT:       416 |       element_type * _M_ptr
// CHECK-NEXT:       424 |       class std::__shared_count<> _M_refcount
// CHECK-NEXT:       424 |         _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT:       432 |   detail::class CG::CGTYPE MCGType
// CHECK-NEXT:       440 |   void * MSrcPtr
// CHECK-NEXT:       448 |   void * MDstPtr
// CHECK-NEXT:       456 |   size_t MLength
// CHECK-NEXT:       464 |   class std::vector<char> MPattern
// CHECK-NEXT:       464 |     struct std::_Vector_base<char, class std::allocator<char> > (base)
// CHECK-NEXT:       464 |       struct std::_Vector_base<char, class std::allocator<char> >::_Vector_impl _M_impl
// CHECK-NEXT:       464 |         class std::allocator<char> (base) (empty)
// CHECK-NEXT:       464 |           class __gnu_cxx::new_allocator<char> (base) (empty)
// CHECK-NEXT:       464 |         struct std::_Vector_base<char, class std::allocator<char> >::_Vector_impl_data (base)
// CHECK:       464 |           pointer _M_start
// CHECK-NEXT:       472 |           pointer _M_finish
// CHECK-NEXT:       480 |           pointer _M_end_of_storage
// CHECK-NEXT:       488 |   class std::unique_ptr<class sycl::detail::HostKernelBase> MHostKernel
// CHECK-NEXT:       488 |     struct std::__uniq_ptr_data<class sycl::detail::HostKernelBase, struct std::default_delete<class sycl::detail::HostKernelBase> > _M_t
// CHECK-NEXT:       488 |       class std::__uniq_ptr_impl<class sycl::detail::HostKernelBase, struct std::default_delete<class sycl::detail::HostKernelBase> > (base)
// CHECK-NEXT:       488 |         class std::tuple<class sycl::detail::HostKernelBase *, struct std::default_delete<class sycl::detail::HostKernelBase> > _M_t
// CHECK-NEXT:       488 |           struct std::_Tuple_impl<0, class sycl::detail::HostKernelBase *, struct std::default_delete<class sycl::detail::HostKernelBase> > (base)
// CHECK-NEXT:       488 |             struct std::_Tuple_impl<1, struct std::default_delete<class sycl::detail::HostKernelBase> > (base) (empty)
// CHECK-NEXT:       488 |               struct std::_Head_base<1, struct std::default_delete<class sycl::detail::HostKernelBase> > (base) (empty)
// CHECK-NEXT:       488 |                 struct std::default_delete<class sycl::detail::HostKernelBase> _M_head_impl (empty)
// CHECK-NEXT:       488 |             struct std::_Head_base<0, class sycl::detail::HostKernelBase *> (base)
// CHECK-NEXT:       488 |               class sycl::detail::HostKernelBase * _M_head_impl
// CHECK-NEXT:       496 |   class std::unique_ptr<class sycl::detail::HostTask> MHostTask
// CHECK-NEXT:       496 |     struct std::__uniq_ptr_data<class sycl::detail::HostTask, struct std::default_delete<class sycl::detail::HostTask> > _M_t
// CHECK-NEXT:       496 |       class std::__uniq_ptr_impl<class sycl::detail::HostTask, struct std::default_delete<class sycl::detail::HostTask> > (base)
// CHECK-NEXT:       496 |         class std::tuple<class sycl::detail::HostTask *, struct std::default_delete<class sycl::detail::HostTask> > _M_t
// CHECK-NEXT:       496 |           struct std::_Tuple_impl<0, class sycl::detail::HostTask *, struct std::default_delete<class sycl::detail::HostTask> > (base)
// CHECK-NEXT:       496 |             struct std::_Tuple_impl<1, struct std::default_delete<class sycl::detail::HostTask> > (base) (empty)
// CHECK-NEXT:       496 |               struct std::_Head_base<1, struct std::default_delete<class sycl::detail::HostTask> > (base) (empty)
// CHECK-NEXT:       496 |                 struct std::default_delete<class sycl::detail::HostTask> _M_head_impl (empty)
// CHECK-NEXT:       496 |             struct std::_Head_base<0, class sycl::detail::HostTask *> (base)
// CHECK-NEXT:       496 |               class sycl::detail::HostTask * _M_head_impl
// CHECK-NEXT:       504 |   class std::unique_ptr<class sycl::detail::InteropTask> MInteropTask
// CHECK-NEXT:       504 |     struct std::__uniq_ptr_data<class sycl::detail::InteropTask, struct std::default_delete<class sycl::detail::InteropTask> > _M_t
// CHECK-NEXT:       504 |       class std::__uniq_ptr_impl<class sycl::detail::InteropTask, struct std::default_delete<class sycl::detail::InteropTask> > (base)
// CHECK-NEXT:       504 |         class std::tuple<class sycl::detail::InteropTask *, struct std::default_delete<class sycl::detail::InteropTask> > _M_t
// CHECK-NEXT:       504 |           struct std::_Tuple_impl<0, class sycl::detail::InteropTask *, struct std::default_delete<class sycl::detail::InteropTask> > (base)
// CHECK-NEXT:       504 |             struct std::_Tuple_impl<1, struct std::default_delete<class sycl::detail::InteropTask> > (base) (empty)
// CHECK-NEXT:       504 |               struct std::_Head_base<1, struct std::default_delete<class sycl::detail::InteropTask> > (base) (empty)
// CHECK-NEXT:       504 |                 struct std::default_delete<class sycl::detail::InteropTask> _M_head_impl (empty)
// CHECK-NEXT:       504 |             struct std::_Head_base<0, class sycl::detail::InteropTask *> (base)
// CHECK-NEXT:       504 |               class sycl::detail::InteropTask * _M_head_impl
// CHECK-NEXT:       512 |   class std::vector<class std::shared_ptr<class sycl::detail::event_impl> > MEventsWaitWithBarrier
// CHECK-NEXT:       512 |     struct std::_Vector_base<class std::shared_ptr<class sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::event_impl> > > (base)
// CHECK-NEXT:       512 |       struct std::_Vector_base<class std::shared_ptr<class sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::event_impl> > >::_Vector_impl _M_impl
// CHECK-NEXT:       512 |         class std::allocator<class std::shared_ptr<class sycl::detail::event_impl> > (base) (empty)
// CHECK-NEXT:       512 |           class __gnu_cxx::new_allocator<class std::shared_ptr<class sycl::detail::event_impl> > (base) (empty)
// CHECK-NEXT:       512 |         struct std::_Vector_base<class std::shared_ptr<class sycl::detail::event_impl>, class std::allocator<class std::shared_ptr<class sycl::detail::event_impl> > >::_Vector_impl_data (base)
// CHECK:       512 |           pointer _M_start
// CHECK-NEXT:       520 |           pointer _M_finish
// CHECK-NEXT:       528 |           pointer _M_end_of_storage
// CHECK-NEXT:       536 |   class std::shared_ptr<class sycl::ext::oneapi::experimental::detail::graph_impl> MGraph
// CHECK-NEXT:       536 |     class std::__shared_ptr<class sycl::ext::oneapi::experimental::detail::graph_impl> (base)
// CHECK-NEXT:       536 |       class std::__shared_ptr_access<class sycl::ext::oneapi::experimental::detail::graph_impl, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT:       536 |       element_type * _M_ptr
// CHECK-NEXT:       544 |       class std::__shared_count<> _M_refcount
// CHECK-NEXT:       544 |         _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT:       552 |   class std::shared_ptr<class sycl::ext::oneapi::experimental::detail::exec_graph_impl> MExecGraph
// CHECK-NEXT:       552 |     class std::__shared_ptr<class sycl::ext::oneapi::experimental::detail::exec_graph_impl> (base)
// CHECK-NEXT:       552 |       class std::__shared_ptr_access<class sycl::ext::oneapi::experimental::detail::exec_graph_impl, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT:       552 |       element_type * _M_ptr
// CHECK-NEXT:       560 |       class std::__shared_count<> _M_refcount
// CHECK-NEXT:       560 |         _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT:       568 |   class std::shared_ptr<class sycl::ext::oneapi::experimental::detail::node_impl> MSubgraphNode
// CHECK-NEXT:       568 |     class std::__shared_ptr<class sycl::ext::oneapi::experimental::detail::node_impl> (base)
// CHECK-NEXT:       568 |       class std::__shared_ptr_access<class sycl::ext::oneapi::experimental::detail::node_impl, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT:       568 |       element_type * _M_ptr
// CHECK-NEXT:       576 |       class std::__shared_count<> _M_refcount
// CHECK-NEXT:       576 |         _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT:       584 |   class std::unique_ptr<class sycl::detail::CG> MGraphNodeCG
// CHECK:       584 |     struct std::__uniq_ptr_data<class sycl::detail::CG, struct std::default_delete<class sycl::detail::CG> > _M_t
// CHECK-NEXT:       584 |       class std::__uniq_ptr_impl<class sycl::detail::CG, struct std::default_delete<class sycl::detail::CG> > (base)
// CHECK-NEXT:       584 |         class std::tuple<class sycl::detail::CG *, struct std::default_delete<class sycl::detail::CG> > _M_t
// CHECK-NEXT:       584 |           struct std::_Tuple_impl<0, class sycl::detail::CG *, struct std::default_delete<class sycl::detail::CG> > (base)
// CHECK-NEXT:       584 |             struct std::_Tuple_impl<1, struct std::default_delete<class sycl::detail::CG> > (base) (empty)
// CHECK-NEXT:       584 |               struct std::_Head_base<1, struct std::default_delete<class sycl::detail::CG> > (base) (empty)
// CHECK-NEXT:       584 |                 struct std::default_delete<class sycl::detail::CG> _M_head_impl (empty)
// CHECK:       584 |             struct std::_Head_base<0, class sycl::detail::CG *> (base)
// CHECK-NEXT:       584 |               class sycl::detail::CG * _M_head_impl
// CHECK-NEXT:       592 |   _Bool MIsHost
// CHECK-NEXT:       600 |   struct sycl::detail::code_location MCodeLoc
// CHECK-NEXT:       600 |     const char * MFileName
// CHECK-NEXT:       608 |     const char * MFunctionName
// CHECK-NEXT:       616 |     unsigned long MLineNo
// CHECK-NEXT:       624 |     unsigned long MColumnNo
// CHECK-NEXT:       632 |   _Bool MIsFinalized
// CHECK-NEXT:       640 |   class sycl::event MLastEvent
// CHECK-NEXT:       640 |     class sycl::detail::OwnerLessBase<class sycl::event> (base) (empty)
// CHECK-NEXT:       640 |     class std::shared_ptr<class sycl::detail::event_impl> impl
// CHECK-NEXT:       640 |       class std::__shared_ptr<class sycl::detail::event_impl> (base)
// CHECK-NEXT:       640 |         class std::__shared_ptr_access<class sycl::detail::event_impl, __gnu_cxx::_S_atomic> (base) (empty)
// CHECK-NEXT:       640 |         element_type * _M_ptr
// CHECK-NEXT:       648 |         class std::__shared_count<> _M_refcount
// CHECK-NEXT:       648 |           _Sp_counted_base<(_Lock_policy)2U> * _M_pi
// CHECK-NEXT:     | [sizeof=656, dsize=656, align=8,
// CHECK-NEXT:     |  nvsize=656, nvalign=8]
