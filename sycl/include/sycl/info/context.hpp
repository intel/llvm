//==----- context.hpp - SYCL context information descriptors --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/info_desc_traits.hpp>
#include <unified-runtime/ur_api.h>

#include <cstdint>
#include <vector>

namespace sycl {
inline namespace _V1 {

class device;
class platform;
enum class memory_scope;
enum class memory_order;

namespace info {
// A.2 Context information desctiptors
namespace context {
template <ur_context_info_t UrCode>
using context_traits =
    sycl::detail::ur_traits_base<sycl::detail::info_class::context, UrCode>;
using context_runtime_traits =
    sycl::detail::rt_traits_base<sycl::detail::info_class::context>;

struct reference_count : context_traits<UR_CONTEXT_INFO_REFERENCE_COUNT> {
  using return_type = uint32_t;
};
struct platform : context_runtime_traits {
  using return_type = sycl::platform;
};
struct devices : context_traits<UR_CONTEXT_INFO_DEVICES> {
  using return_type = std::vector<sycl::device>;
};
struct atomic_memory_order_capabilities : context_runtime_traits {
  using return_type = std::vector<sycl::memory_order>;
};
struct atomic_memory_scope_capabilities : context_runtime_traits {
  using return_type = std::vector<sycl::memory_scope>;
};
struct atomic_fence_order_capabilities : context_runtime_traits {
  using return_type = std::vector<sycl::memory_order>;
};
struct atomic_fence_scope_capabilities : context_runtime_traits {
  using return_type = std::vector<sycl::memory_scope>;
};
} // namespace context
} // namespace info

namespace detail {
// SFINAE predicate confining `context::get_info<T>()` to context traits.
// `return_type` alias is load-bearing for ABI symbol mangling — keep stable.
template <typename T>
struct is_context_info_desc : is_info_desc_for<T, info_class::context> {};
} // namespace detail
} // namespace _V1
} // namespace sycl
