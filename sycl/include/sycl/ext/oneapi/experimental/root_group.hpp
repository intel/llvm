//==--- root_group.hpp --- SYCL extension for root groups ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/builtins.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <sycl/memory_enums.hpp>
#include <sycl/queue.hpp>

#define SYCL_EXT_ONEAPI_ROOT_GROUP 1

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi::experimental {

namespace info::kernel_queue_specific {
// TODO: Revisit and align with sycl_ext_oneapi_forward_progress extension once
// #7598 is merged.
struct max_num_work_group_sync {
  using return_type = size_t;
};
} // namespace info::kernel_queue_specific

struct use_root_sync_key {
  using value_t = property_value<use_root_sync_key>;
};

inline constexpr use_root_sync_key::value_t use_root_sync;

template <> struct is_property_key<use_root_sync_key> : std::true_type {};

template <> struct detail::PropertyToKind<use_root_sync_key> {
  static constexpr PropKind Kind = PropKind::UseRootSync;
};

template <>
struct detail::IsCompileTimeProperty<use_root_sync_key> : std::true_type {};

template <int Dimensions> class root_group {
public:
  using id_type = id<Dimensions>;
  using range_type = range<Dimensions>;
  using linear_id_type = size_t;
  static constexpr int dimensions = Dimensions;
  static constexpr memory_scope fence_scope = memory_scope::device;

  id<Dimensions> get_group_id() const { return id<Dimensions>{}; };

  id<Dimensions> get_local_id() const { return it.get_global_id(); }

  range<Dimensions> get_group_range() const {
    if constexpr (Dimensions == 3) {
      return range<3>{1, 1, 1};
    } else if constexpr (Dimensions == 2) {
      return range<2>{1, 1};
    } else {
      return range<1>{1};
    }
  }

  range<Dimensions> get_local_range() const { return it.get_global_range(); };

  range<Dimensions> get_max_local_range() const { return get_local_range(); };

  size_t get_group_linear_id() const { return 0; };

  size_t get_local_linear_id() const { return it.get_global_linear_id(); }

  size_t get_group_linear_range() const { return get_group_range().size(); };

  size_t get_local_linear_range() const { return get_local_range().size(); };

  bool leader() const { return get_local_id() == 0; };

private:
  friend root_group<Dimensions>
  nd_item<Dimensions>::ext_oneapi_get_root_group() const;

  root_group(nd_item<Dimensions> it) : it{it} {}

  sycl::nd_item<Dimensions> it;
};

template <int Dimensions>
group<Dimensions> get_child_group(root_group<Dimensions> g) {
  (void)g;
  return this_group<Dimensions>();
}

template <int Dimensions> sub_group get_child_group(group<Dimensions> g) {
  (void)g;
  return this_sub_group();
}

namespace this_kernel {
template <int Dimensions> root_group<Dimensions> get_root_group() {
  return this_nd_item<Dimensions>().ext_oneapi_get_root_group();
}
} // namespace this_kernel

} // namespace ext::oneapi::experimental

template <>
typename ext::oneapi::experimental::info::kernel_queue_specific::
    max_num_work_group_sync::return_type
    kernel::ext_oneapi_get_info<
        ext::oneapi::experimental::info::kernel_queue_specific::
            max_num_work_group_sync>(const queue &q) const {
  // TODO: query the backend to return a value >= 1.
  return 1;
}

template <int dimensions>
void group_barrier(ext::oneapi::experimental::root_group<dimensions> G,
                   memory_scope FenceScope = decltype(G)::fence_scope) {
  (void)G;
  (void)FenceScope;
#ifdef __SYCL_DEVICE_ONLY__
  // TODO: Change __spv::Scope::Workgroup to __spv::Scope::Device once backends
  // support device scope. __spv::Scope::Workgroup is only valid when
  // max_num_work_group_sync is 1, so that all work items in a root group will
  // also be in the same work group.
  __spirv_ControlBarrier(__spv::Scope::Workgroup, __spv::Scope::Workgroup,
                         __spv::MemorySemanticsMask::SubgroupMemory |
                             __spv::MemorySemanticsMask::WorkgroupMemory |
                             __spv::MemorySemanticsMask::CrossWorkgroupMemory);
#else
  throw sycl::runtime_error("Barriers are not supported on host device",
                            PI_ERROR_INVALID_DEVICE);
#endif
}

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
