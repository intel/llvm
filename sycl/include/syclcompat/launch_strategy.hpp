/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Part of the LLVM Project, under the Apache License v2.0 with LLVM
 *  Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCL compatibility extension
 *
 *  launch.hpp
 *
 *  Description:
 *    launch functionality for the SYCL compatibility extension
 **************************************************************************/

#pragma once

// #include <sycl/accessor.hpp>
#include "sycl/ext/oneapi/properties/properties.hpp"
#include <sycl/event.hpp>
#include <sycl/nd_range.hpp>
#include <sycl/queue.hpp>
#include <sycl/range.hpp>
// #include <sycl/reduction.hpp>

#include <syclcompat/device.hpp>
#include <syclcompat/dims.hpp>
#include <syclcompat/traits.hpp>

namespace syclcompat {
namespace experimental {

namespace sycl_exp = sycl::ext::oneapi::experimental;

// launch_strategy is constructed by the user & passed to `compat_exp::launch`
template <typename RangeT, typename KProps, typename LProps>
struct launch_strategy{
  static_assert(sycl_exp::is_property_list_v<KProps>);
  static_assert(sycl_exp::is_property_list_v<LProps>);

  launch_strategy(RangeT range, KProps kprops, LProps lprops, size_t lmem_size)
      : range{range}, kernel_properties{kprops}, launch_properties{lprops},
        local_mem_size{lmem_size} {}
  RangeT range;
  KProps kernel_properties;
  LProps launch_properties;
  size_t local_mem_size;
};

// TODO: std::true_type etc inheritance to create `is_launch_strategy_v`
// TODO: assert RangeT is nd_range or range
// TODO: ctors taking `dim3`, `dim3, dim3`


namespace detail {

// Atharva's stuff
//====================================================================
template <auto KernelFunc, typename tuple, std::size_t... I>
__attribute__((always_inline)) inline void
run_kernel(tuple args, std::index_sequence<I...>) {
  KernelFunc(std::get<I>(args)...);
}

template <auto KernelFunc, typename tuple>
__attribute__((always_inline)) inline void run_kernel(tuple args) {
  auto indices = std::make_index_sequence<std::tuple_size_v<tuple>>{};
  run_kernel<KernelFunc>(args, indices);
}

// TODO: This is a good basis but we need to:
// - Extract SubgroupSize & generalize impl
// - Get local mem working
template <int SubgroupSize, auto KernelFunc, typename... Args>
struct KernelFunctor {
  KernelFunctor(Args... args) : argument_tuple(std::make_tuple(args...)) {}

  auto get(sycl_exp::properties_tag) {
    return sycl_exp::properties{
        sycl_exp::sub_group_size<SubgroupSize>};
  }

  __attribute__((always_inline)) inline void
  operator()(sycl::nd_item<3> it) const {
    run_kernel<KernelFunc>(argument_tuple);
  }

  std::tuple<Args...> argument_tuple;
};
//====================================================================

template <auto F, typename LaunchStrategy, typename... Args>
sycl::event launch(LaunchStrategy launch_strategy, sycl::queue q, Args... args) {
  static_assert(syclcompat::args_compatible<F, Args...>);
  return sycl::event{};
}

} // namespace detail


template <auto F, typename LaunchStrategy, typename... Args>
std::enable_if_t<syclcompat::args_compatible<F, Args...>, sycl::event>
launch(LaunchStrategy launch_strategy, sycl::queue q, Args... args) {
  return detail::launch<F>(launch_strategy, q, args...);
}


template <auto F, typename LaunchStrategy, typename... Args>
std::enable_if_t<syclcompat::args_compatible<F, Args...>, sycl::event>
launch(LaunchStrategy launch_strategy, Args... args) {
  return launch<F>(launch_strategy, get_default_queue(), args...);
}


} // namespace experimental
} // namespace syclcompat
