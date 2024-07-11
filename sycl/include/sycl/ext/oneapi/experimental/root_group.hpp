//==--- root_group.hpp --- SYCL extension for root groups ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/spirv.hpp>
#include <sycl/ext/oneapi/experimental/use_root_sync_prop.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/group.hpp>
#include <sycl/memory_enums.hpp>
#include <sycl/nd_item.hpp>
#include <sycl/sub_group.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/ext/oneapi/functional.hpp>
#endif

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

namespace info::kernel_queue_specific {
// TODO: Revisit and align with sycl_ext_oneapi_forward_progress extension once
// #7598 is merged.
struct max_num_work_group_sync {
  using return_type = size_t;
};
} // namespace info::kernel_queue_specific

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

namespace this_work_item {
template <int Dimensions> root_group<Dimensions> get_root_group() {
  return sycl::ext::oneapi::this_work_item::get_nd_item<Dimensions>()
      .ext_oneapi_get_root_group();
}
} // namespace this_work_item

namespace this_kernel {
template <int Dimensions>
__SYCL_DEPRECATED(
    "use sycl::ext::oneapi::experimental::this_work_item::get_root_group() "
    "instead")
root_group<Dimensions> get_root_group() {
  this_work_item::get_root_group<Dimensions>();
}
} // namespace this_kernel

} // namespace ext::oneapi::experimental

template <int dimensions>
void group_barrier(ext::oneapi::experimental::root_group<dimensions> G,
                   memory_scope FenceScope = decltype(G)::fence_scope) {
#ifdef __SYCL_DEVICE_ONLY__
  // Root group barrier synchronizes using a work group barrier if there's only
  // one work group. This allows backends to ignore the ControlBarrier with
  // Device scope if their maximum number of work groups is 1. This is a
  // workaround that's not intended to reduce the bar for SPIR-V modules
  // acceptance, but rather make a pessimistic case work until we have full
  // support for the device barrier built-in from backends.
  const auto ChildGroup = ext::oneapi::experimental::this_group<dimensions>();
  if (ChildGroup.get_group_linear_range() == 1) {
    group_barrier(ChildGroup);
  } else {
    detail::spirv::ControlBarrier(G, FenceScope, memory_order::seq_cst);
  }
#else
  (void)G;
  (void)FenceScope;
  throw sycl::exception(make_error_code(errc::runtime),
                        "Barriers are not supported on host");
#endif
}

} // namespace _V1
} // namespace sycl
