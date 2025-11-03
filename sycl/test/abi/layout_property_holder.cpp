// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// RUN: %clangxx -fsycl -fsycl-device-only -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s

// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <sycl/detail/kernel_launch_helper.hpp>

// CHECK: *** Dumping AST Record Layout
// CHECK: 0 | struct sycl::detail::PropsHolder<struct sycl::ext::oneapi::experimental::work_group_scratch_size, struct sycl::ext::intel::experimental::cache_config, struct sycl::ext::oneapi::experimental::use_root_sync_key, struct sycl::ext::oneapi::experimental::work_group_progress_key, struct sycl::ext::oneapi::experimental::sub_group_progress_key, struct sycl::ext::oneapi::experimental::work_item_progress_key, struct sycl::ext::oneapi::experimental::cuda::cluster_size<1>, struct sycl::ext::oneapi::experimental::cuda::cluster_size<2>, struct sycl::ext::oneapi::experimental::cuda::cluster_size<3> >
// CHECK: 0 |   struct sycl::detail::MarshalledProperty<struct sycl::ext::oneapi::experimental::work_group_scratch_size> (base)
// CHECK: 0 |     class std::optional<struct sycl::ext::oneapi::experimental::work_group_scratch_size> MProperty
// CHECK: 0 |       struct std::_Optional_base<struct sycl::ext::oneapi::experimental::work_group_scratch_size> (base)
// CHECK: 0 |         class std::_Optional_base_impl<struct sycl::ext::oneapi::experimental::work_group_scratch_size, struct std::_Optional_base<struct sycl::ext::oneapi::experimental::work_group_scratch_size> > (base) (empty)
// CHECK: 0 |         struct std::_Optional_payload<struct sycl::ext::oneapi::experimental::work_group_scratch_size> _M_payload
// CHECK: 0 |           struct std::_Optional_payload_base<struct sycl::ext::oneapi::experimental::work_group_scratch_size> (base)
// CHECK: 0 |             union std::_Optional_payload_base<struct sycl::ext::oneapi::experimental::work_group_scratch_size>::_Storage<struct sycl::ext::oneapi::experimental::work_group_scratch_size> _M_payload
// CHECK: 0 |               struct std::_Optional_payload_base<struct sycl::ext::oneapi::experimental::work_group_scratch_size>::_Empty_byte _M_empty (empty)
// CHECK: 0 |               struct sycl::ext::oneapi::experimental::work_group_scratch_size _M_value
// CHECK: 0 |                 struct sycl::ext::oneapi::experimental::detail::run_time_property_key<struct sycl::ext::oneapi::experimental::work_group_scratch_size, sycl::ext::oneapi::experimental::detail::WorkGroupScratchSize> (base) (empty)
// CHECK: 0 |                   struct sycl::ext::oneapi::experimental::detail::property_key_base_tag (base) (empty)
// CHECK: 0 |                   struct sycl::ext::oneapi::experimental::detail::property_base<struct sycl::ext::oneapi::experimental::work_group_scratch_size, sycl::ext::oneapi::experimental::detail::WorkGroupScratchSize> (base) (empty)
// CHECK: 0 |                     struct sycl::ext::oneapi::experimental::detail::property_key_tag<struct sycl::ext::oneapi::experimental::work_group_scratch_size> (base) (empty)
// CHECK: 0 |                       struct sycl::ext::oneapi::experimental::detail::property_tag (base) (empty)
// CHECK: 0 |                 size_t size
// CHECK: 8 |             _Bool _M_engaged
// CHECK: 0 |       struct std::_Enable_copy_move<true, true, true, true, class std::optional<struct sycl::ext::oneapi::experimental::work_group_scratch_size> > (base) (empty)
// CHECK: 16 |   struct sycl::detail::MarshalledProperty<struct sycl::ext::intel::experimental::cache_config> (base)
// CHECK: 16 |     class std::optional<struct sycl::ext::intel::experimental::cache_config> MProperty
// CHECK: 16 |       struct std::_Optional_base<struct sycl::ext::intel::experimental::cache_config> (base)
// CHECK: 16 |         class std::_Optional_base_impl<struct sycl::ext::intel::experimental::cache_config, struct std::_Optional_base<struct sycl::ext::intel::experimental::cache_config> > (base) (empty)
// CHECK: 16 |         struct std::_Optional_payload<struct sycl::ext::intel::experimental::cache_config> _M_payload
// CHECK: 16 |           struct std::_Optional_payload_base<struct sycl::ext::intel::experimental::cache_config> (base)
// CHECK: 16 |             union std::_Optional_payload_base<struct sycl::ext::intel::experimental::cache_config>::_Storage<struct sycl::ext::intel::experimental::cache_config> _M_payload
// CHECK: 16 |               struct std::_Optional_payload_base<struct sycl::ext::intel::experimental::cache_config>::_Empty_byte _M_empty (empty)
// CHECK: 16 |               struct sycl::ext::intel::experimental::cache_config _M_value
// CHECK: 16 |                 struct sycl::ext::oneapi::experimental::detail::run_time_property_key<struct sycl::ext::intel::experimental::cache_config, sycl::ext::oneapi::experimental::detail::CacheConfig> (base) (empty)
// CHECK: 16 |                   struct sycl::ext::oneapi::experimental::detail::property_key_base_tag (base) (empty)
// CHECK: 16 |                   struct sycl::ext::oneapi::experimental::detail::property_base<struct sycl::ext::intel::experimental::cache_config, sycl::ext::oneapi::experimental::detail::CacheConfig> (base) (empty)
// CHECK: 16 |                     struct sycl::ext::oneapi::experimental::detail::property_key_tag<struct sycl::ext::intel::experimental::cache_config> (base) (empty)
// CHECK: 16 |                       struct sycl::ext::oneapi::experimental::detail::property_tag (base) (empty)
// CHECK: 16 |                 cache_config_enum value
// CHECK: 18 |             _Bool _M_engaged
// CHECK: 16 |       struct std::_Enable_copy_move<true, true, true, true, class std::optional<struct sycl::ext::intel::experimental::cache_config> > (base) (empty)
// CHECK: 20 |   struct sycl::detail::MarshalledProperty<struct sycl::ext::oneapi::experimental::use_root_sync_key> (base)
// CHECK: 20 |     _Bool MPresent
// CHECK: 24 |   struct sycl::detail::MarshalledProperty<struct sycl::ext::oneapi::experimental::work_group_progress_key> (base)
// CHECK: 24 |     class std::optional<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> MFPGuarantee
// CHECK: 24 |       struct std::_Optional_base<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> (base)
// CHECK: 24 |         class std::_Optional_base_impl<enum sycl::ext::oneapi::experimental::forward_progress_guarantee, struct std::_Optional_base<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> > (base) (empty)
// CHECK: 24 |         struct std::_Optional_payload<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> _M_payload
// CHECK: 24 |           struct std::_Optional_payload_base<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> (base)
// CHECK: 24 |             union std::_Optional_payload_base<enum sycl::ext::oneapi::experimental::forward_progress_guarantee>::_Storage<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> _M_payload
// CHECK: 24 |               struct std::_Optional_payload_base<enum sycl::ext::oneapi::experimental::forward_progress_guarantee>::_Empty_byte _M_empty (empty)
// CHECK: 24 |               enum sycl::ext::oneapi::experimental::forward_progress_guarantee _M_value
// CHECK: 28 |             _Bool _M_engaged
// CHECK: 24 |       struct std::_Enable_copy_move<true, true, true, true, class std::optional<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> > (base) (empty)
// CHECK: 32 |     class std::optional<enum sycl::ext::oneapi::experimental::execution_scope> MFPCoordinationScope
// CHECK: 32 |       struct std::_Optional_base<enum sycl::ext::oneapi::experimental::execution_scope> (base)
// CHECK: 32 |         class std::_Optional_base_impl<enum sycl::ext::oneapi::experimental::execution_scope, struct std::_Optional_base<enum sycl::ext::oneapi::experimental::execution_scope> > (base) (empty)
// CHECK: 32 |         struct std::_Optional_payload<enum sycl::ext::oneapi::experimental::execution_scope> _M_payload
// CHECK: 32 |           struct std::_Optional_payload_base<enum sycl::ext::oneapi::experimental::execution_scope> (base)
// CHECK: 32 |             union std::_Optional_payload_base<enum sycl::ext::oneapi::experimental::execution_scope>::_Storage<enum sycl::ext::oneapi::experimental::execution_scope> _M_payload
// CHECK: 32 |               struct std::_Optional_payload_base<enum sycl::ext::oneapi::experimental::execution_scope>::_Empty_byte _M_empty (empty)
// CHECK: 32 |               enum sycl::ext::oneapi::experimental::execution_scope _M_value
// CHECK: 36 |             _Bool _M_engaged
// CHECK: 32 |       struct std::_Enable_copy_move<true, true, true, true, class std::optional<enum sycl::ext::oneapi::experimental::execution_scope> > (base) (empty)
// CHECK: 40 |   struct sycl::detail::MarshalledProperty<struct sycl::ext::oneapi::experimental::sub_group_progress_key> (base)
// CHECK: 40 |     class std::optional<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> MFPGuarantee
// CHECK: 40 |       struct std::_Optional_base<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> (base)
// CHECK: 40 |         class std::_Optional_base_impl<enum sycl::ext::oneapi::experimental::forward_progress_guarantee, struct std::_Optional_base<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> > (base) (empty)
// CHECK: 40 |         struct std::_Optional_payload<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> _M_payload
// CHECK: 40 |           struct std::_Optional_payload_base<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> (base)
// CHECK: 40 |             union std::_Optional_payload_base<enum sycl::ext::oneapi::experimental::forward_progress_guarantee>::_Storage<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> _M_payload
// CHECK: 40 |               struct std::_Optional_payload_base<enum sycl::ext::oneapi::experimental::forward_progress_guarantee>::_Empty_byte _M_empty (empty)
// CHECK: 40 |               enum sycl::ext::oneapi::experimental::forward_progress_guarantee _M_value
// CHECK: 44 |             _Bool _M_engaged
// CHECK: 40 |       struct std::_Enable_copy_move<true, true, true, true, class std::optional<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> > (base) (empty)
// CHECK: 48 |     class std::optional<enum sycl::ext::oneapi::experimental::execution_scope> MFPCoordinationScope
// CHECK: 48 |       struct std::_Optional_base<enum sycl::ext::oneapi::experimental::execution_scope> (base)
// CHECK: 48 |         class std::_Optional_base_impl<enum sycl::ext::oneapi::experimental::execution_scope, struct std::_Optional_base<enum sycl::ext::oneapi::experimental::execution_scope> > (base) (empty)
// CHECK: 48 |         struct std::_Optional_payload<enum sycl::ext::oneapi::experimental::execution_scope> _M_payload
// CHECK: 48 |           struct std::_Optional_payload_base<enum sycl::ext::oneapi::experimental::execution_scope> (base)
// CHECK: 48 |             union std::_Optional_payload_base<enum sycl::ext::oneapi::experimental::execution_scope>::_Storage<enum sycl::ext::oneapi::experimental::execution_scope> _M_payload
// CHECK: 48 |               struct std::_Optional_payload_base<enum sycl::ext::oneapi::experimental::execution_scope>::_Empty_byte _M_empty (empty)
// CHECK: 48 |               enum sycl::ext::oneapi::experimental::execution_scope _M_value
// CHECK: 52 |             _Bool _M_engaged
// CHECK: 48 |       struct std::_Enable_copy_move<true, true, true, true, class std::optional<enum sycl::ext::oneapi::experimental::execution_scope> > (base) (empty)
// CHECK: 56 |   struct sycl::detail::MarshalledProperty<struct sycl::ext::oneapi::experimental::work_item_progress_key> (base)
// CHECK: 56 |     class std::optional<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> MFPGuarantee
// CHECK: 56 |       struct std::_Optional_base<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> (base)
// CHECK: 56 |         class std::_Optional_base_impl<enum sycl::ext::oneapi::experimental::forward_progress_guarantee, struct std::_Optional_base<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> > (base) (empty)
// CHECK: 56 |         struct std::_Optional_payload<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> _M_payload
// CHECK: 56 |           struct std::_Optional_payload_base<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> (base)
// CHECK: 56 |             union std::_Optional_payload_base<enum sycl::ext::oneapi::experimental::forward_progress_guarantee>::_Storage<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> _M_payload
// CHECK: 56 |               struct std::_Optional_payload_base<enum sycl::ext::oneapi::experimental::forward_progress_guarantee>::_Empty_byte _M_empty (empty)
// CHECK: 56 |               enum sycl::ext::oneapi::experimental::forward_progress_guarantee _M_value
// CHECK: 60 |             _Bool _M_engaged
// CHECK: 56 |       struct std::_Enable_copy_move<true, true, true, true, class std::optional<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> > (base) (empty)
// CHECK: 64 |     class std::optional<enum sycl::ext::oneapi::experimental::execution_scope> MFPCoordinationScope
// CHECK: 64 |       struct std::_Optional_base<enum sycl::ext::oneapi::experimental::execution_scope> (base)
// CHECK: 64 |         class std::_Optional_base_impl<enum sycl::ext::oneapi::experimental::execution_scope, struct std::_Optional_base<enum sycl::ext::oneapi::experimental::execution_scope> > (base) (empty)
// CHECK: 64 |         struct std::_Optional_payload<enum sycl::ext::oneapi::experimental::execution_scope> _M_payload
// CHECK: 64 |           struct std::_Optional_payload_base<enum sycl::ext::oneapi::experimental::execution_scope> (base)
// CHECK: 64 |             union std::_Optional_payload_base<enum sycl::ext::oneapi::experimental::execution_scope>::_Storage<enum sycl::ext::oneapi::experimental::execution_scope> _M_payload
// CHECK: 64 |               struct std::_Optional_payload_base<enum sycl::ext::oneapi::experimental::execution_scope>::_Empty_byte _M_empty (empty)
// CHECK: 64 |               enum sycl::ext::oneapi::experimental::execution_scope _M_value
// CHECK: 68 |             _Bool _M_engaged
// CHECK: 64 |       struct std::_Enable_copy_move<true, true, true, true, class std::optional<enum sycl::ext::oneapi::experimental::execution_scope> > (base) (empty)
// CHECK: 72 |   struct sycl::detail::MarshalledProperty<struct sycl::ext::oneapi::experimental::cuda::cluster_size<1> > (base)
// CHECK: 72 |     class std::optional<struct sycl::ext::oneapi::experimental::cuda::cluster_size<1> > MProperty
// CHECK: 72 |       struct std::_Optional_base<struct sycl::ext::oneapi::experimental::cuda::cluster_size<1> > (base)
// CHECK: 72 |         class std::_Optional_base_impl<struct sycl::ext::oneapi::experimental::cuda::cluster_size<1>, struct std::_Optional_base<struct sycl::ext::oneapi::experimental::cuda::cluster_size<1> > > (base) (empty)
// CHECK: 72 |         struct std::_Optional_payload<struct sycl::ext::oneapi::experimental::cuda::cluster_size<1> > _M_payload
// CHECK: 72 |           struct std::_Optional_payload_base<struct sycl::ext::oneapi::experimental::cuda::cluster_size<1> > (base)
// CHECK: 72 |             union std::_Optional_payload_base<struct sycl::ext::oneapi::experimental::cuda::cluster_size<1> >::_Storage<struct sycl::ext::oneapi::experimental::cuda::cluster_size<1> > _M_payload
// CHECK: 72 |               struct std::_Optional_payload_base<struct sycl::ext::oneapi::experimental::cuda::cluster_size<1> >::_Empty_byte _M_empty (empty)
// CHECK: 72 |               struct sycl::ext::oneapi::experimental::cuda::cluster_size<1> _M_value
// CHECK: 72 |                 struct sycl::ext::oneapi::experimental::detail::run_time_property_key<struct sycl::ext::oneapi::experimental::cuda::cluster_size<1>, sycl::ext::oneapi::experimental::detail::ClusterLaunch> (base) (empty)
// CHECK: 72 |                   struct sycl::ext::oneapi::experimental::detail::property_key_base_tag (base) (empty)
// CHECK: 72 |                   struct sycl::ext::oneapi::experimental::detail::property_base<struct sycl::ext::oneapi::experimental::cuda::cluster_size<1>, sycl::ext::oneapi::experimental::detail::ClusterLaunch> (base) (empty)
// CHECK: 72 |                     struct sycl::ext::oneapi::experimental::detail::property_key_tag<struct sycl::ext::oneapi::experimental::cuda::cluster_size<1> > (base) (empty)
// CHECK: 72 |                       struct sycl::ext::oneapi::experimental::detail::property_tag (base) (empty)
// CHECK: 72 |                 class sycl::range<> size
// CHECK: 72 |                   class sycl::detail::array<> (base)
// CHECK: 72 |                     size_t[1] common_array
// CHECK: 80 |             _Bool _M_engaged
// CHECK: 72 |       struct std::_Enable_copy_move<true, true, true, true, class std::optional<struct sycl::ext::oneapi::experimental::cuda::cluster_size<1> > > (base) (empty)
// CHECK: 88 |   struct sycl::detail::MarshalledProperty<struct sycl::ext::oneapi::experimental::cuda::cluster_size<2> > (base)
// CHECK: 88 |     class std::optional<struct sycl::ext::oneapi::experimental::cuda::cluster_size<2> > MProperty
// CHECK: 88 |       struct std::_Optional_base<struct sycl::ext::oneapi::experimental::cuda::cluster_size<2> > (base)
// CHECK: 88 |         class std::_Optional_base_impl<struct sycl::ext::oneapi::experimental::cuda::cluster_size<2>, struct std::_Optional_base<struct sycl::ext::oneapi::experimental::cuda::cluster_size<2> > > (base) (empty)
// CHECK: 88 |         struct std::_Optional_payload<struct sycl::ext::oneapi::experimental::cuda::cluster_size<2> > _M_payload
// CHECK: 88 |           struct std::_Optional_payload_base<struct sycl::ext::oneapi::experimental::cuda::cluster_size<2> > (base)
// CHECK: 88 |             union std::_Optional_payload_base<struct sycl::ext::oneapi::experimental::cuda::cluster_size<2> >::_Storage<struct sycl::ext::oneapi::experimental::cuda::cluster_size<2> > _M_payload
// CHECK: 88 |               struct std::_Optional_payload_base<struct sycl::ext::oneapi::experimental::cuda::cluster_size<2> >::_Empty_byte _M_empty (empty)
// CHECK: 88 |               struct sycl::ext::oneapi::experimental::cuda::cluster_size<2> _M_value
// CHECK: 88 |                 struct sycl::ext::oneapi::experimental::detail::run_time_property_key<struct sycl::ext::oneapi::experimental::cuda::cluster_size<2>, sycl::ext::oneapi::experimental::detail::ClusterLaunch> (base) (empty)
// CHECK: 88 |                   struct sycl::ext::oneapi::experimental::detail::property_key_base_tag (base) (empty)
// CHECK: 88 |                   struct sycl::ext::oneapi::experimental::detail::property_base<struct sycl::ext::oneapi::experimental::cuda::cluster_size<2>, sycl::ext::oneapi::experimental::detail::ClusterLaunch> (base) (empty)
// CHECK: 88 |                     struct sycl::ext::oneapi::experimental::detail::property_key_tag<struct sycl::ext::oneapi::experimental::cuda::cluster_size<2> > (base) (empty)
// CHECK: 88 |                       struct sycl::ext::oneapi::experimental::detail::property_tag (base) (empty)
// CHECK: 88 |                 class sycl::range<2> size
// CHECK: 88 |                   class sycl::detail::array<2> (base)
// CHECK: 88 |                     size_t[2] common_array
// CHECK: 104 |             _Bool _M_engaged
// CHECK: 88 |       struct std::_Enable_copy_move<true, true, true, true, class std::optional<struct sycl::ext::oneapi::experimental::cuda::cluster_size<2> > > (base) (empty)
// CHECK: 112 |   struct sycl::detail::MarshalledProperty<struct sycl::ext::oneapi::experimental::cuda::cluster_size<3> > (base)
// CHECK: 112 |     class std::optional<struct sycl::ext::oneapi::experimental::cuda::cluster_size<3> > MProperty
// CHECK: 112 |       struct std::_Optional_base<struct sycl::ext::oneapi::experimental::cuda::cluster_size<3> > (base)
// CHECK: 112 |         class std::_Optional_base_impl<struct sycl::ext::oneapi::experimental::cuda::cluster_size<3>, struct std::_Optional_base<struct sycl::ext::oneapi::experimental::cuda::cluster_size<3> > > (base) (empty)
// CHECK: 112 |         struct std::_Optional_payload<struct sycl::ext::oneapi::experimental::cuda::cluster_size<3> > _M_payload
// CHECK: 112 |           struct std::_Optional_payload_base<struct sycl::ext::oneapi::experimental::cuda::cluster_size<3> > (base)
// CHECK: 112 |             union std::_Optional_payload_base<struct sycl::ext::oneapi::experimental::cuda::cluster_size<3> >::_Storage<struct sycl::ext::oneapi::experimental::cuda::cluster_size<3> > _M_payload
// CHECK: 112 |               struct std::_Optional_payload_base<struct sycl::ext::oneapi::experimental::cuda::cluster_size<3> >::_Empty_byte _M_empty (empty)
// CHECK: 112 |               struct sycl::ext::oneapi::experimental::cuda::cluster_size<3> _M_value
// CHECK: 112 |                 struct sycl::ext::oneapi::experimental::detail::run_time_property_key<struct sycl::ext::oneapi::experimental::cuda::cluster_size<3>, sycl::ext::oneapi::experimental::detail::ClusterLaunch> (base) (empty)
// CHECK: 112 |                   struct sycl::ext::oneapi::experimental::detail::property_key_base_tag (base) (empty)
// CHECK: 112 |                   struct sycl::ext::oneapi::experimental::detail::property_base<struct sycl::ext::oneapi::experimental::cuda::cluster_size<3>, sycl::ext::oneapi::experimental::detail::ClusterLaunch> (base) (empty)
// CHECK: 112 |                     struct sycl::ext::oneapi::experimental::detail::property_key_tag<struct sycl::ext::oneapi::experimental::cuda::cluster_size<3> > (base) (empty)
// CHECK: 112 |                       struct sycl::ext::oneapi::experimental::detail::property_tag (base) (empty)
// CHECK: 112 |                 class sycl::range<3> size
// CHECK: 112 |                   class sycl::detail::array<3> (base)
// CHECK: 112 |                     size_t[3] common_array
// CHECK: 136 |             _Bool _M_engaged
// CHECK: 112 |       struct std::_Enable_copy_move<true, true, true, true, class std::optional<struct sycl::ext::oneapi::experimental::cuda::cluster_size<3> > > (base) (empty)
// CHECK: 144 |   _Bool MEmpty
// CHECK: | [sizeof=152, dsize=145, align=8,
// CHECK: |  nvsize=145, nvalign=8]

SYCL_EXTERNAL void foo(sycl::detail::KernelPropertyHolderStructTy prop) {}
