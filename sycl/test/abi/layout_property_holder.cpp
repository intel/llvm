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
// CHECK: 16 |   struct sycl::detail::MarshalledProperty<struct sycl::ext::intel::experimental::cache_config> (base)
// CHECK: 16 |     class std::optional<struct sycl::ext::intel::experimental::cache_config> MProperty
// CHECK: 20 |   struct sycl::detail::MarshalledProperty<struct sycl::ext::oneapi::experimental::use_root_sync_key> (base)
// CHECK: 20 |     _Bool MPresent
// CHECK: 24 |   struct sycl::detail::MarshalledProperty<struct sycl::ext::oneapi::experimental::work_group_progress_key> (base)
// CHECK: 24 |     class std::optional<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> MFPGuarantee
// CHECK: 32 |     class std::optional<enum sycl::ext::oneapi::experimental::execution_scope> MFPCoordinationScope
// CHECK: 40 |   struct sycl::detail::MarshalledProperty<struct sycl::ext::oneapi::experimental::sub_group_progress_key> (base)
// CHECK: 40 |     class std::optional<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> MFPGuarantee
// CHECK: 48 |     class std::optional<enum sycl::ext::oneapi::experimental::execution_scope> MFPCoordinationScope
// CHECK: 56 |   struct sycl::detail::MarshalledProperty<struct sycl::ext::oneapi::experimental::work_item_progress_key> (base)
// CHECK: 56 |     class std::optional<enum sycl::ext::oneapi::experimental::forward_progress_guarantee> MFPGuarantee
// CHECK: 64 |     class std::optional<enum sycl::ext::oneapi::experimental::execution_scope> MFPCoordinationScope
// CHECK: 72 |   struct sycl::detail::MarshalledProperty<struct sycl::ext::oneapi::experimental::cuda::cluster_size<1> > (base)
// CHECK: 72 |     class std::optional<struct sycl::ext::oneapi::experimental::cuda::cluster_size<1> > MProperty
// CHECK: 88 |   struct sycl::detail::MarshalledProperty<struct sycl::ext::oneapi::experimental::cuda::cluster_size<2> > (base)
// CHECK: 88 |     class std::optional<struct sycl::ext::oneapi::experimental::cuda::cluster_size<2> > MProperty
// CHECK: 112 |   struct sycl::detail::MarshalledProperty<struct sycl::ext::oneapi::experimental::cuda::cluster_size<3> > (base)
// CHECK: 112 |     class std::optional<struct sycl::ext::oneapi::experimental::cuda::cluster_size<3> > MProperty
// CHECK: 144 |   _Bool MEmpty
// CHECK: | [sizeof=152, dsize=145, align=8,
// CHECK: |  nvsize=145, nvalign=8]

SYCL_EXTERNAL void foo(sycl::detail::KernelPropertyHolderStructTy prop) {}
