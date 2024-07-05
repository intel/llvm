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

#include "sycl/ext/oneapi/properties/properties.hpp"
#include "sycl/ext/oneapi/experimental/enqueue_functions.hpp"
#include <sycl/event.hpp>
#include <sycl/nd_range.hpp>
#include <sycl/queue.hpp>
#include <sycl/range.hpp>

#include <syclcompat/device.hpp>
#include <syclcompat/dims.hpp>
#include <syclcompat/traits.hpp>
#include <syclcompat/defs.hpp>

namespace syclcompat {
namespace experimental {

namespace sycl_exp = sycl::ext::oneapi::experimental;

// launch_policy is constructed by the user & passed to `compat_exp::launch`
template <typename Range, typename KProps, typename LProps>
struct launch_policy {
  static_assert(sycl_exp::is_property_list_v<KProps>);
  static_assert(sycl_exp::is_property_list_v<LProps>);
  static_assert(syclcompat::detail::is_range_or_nd_range_v<Range>);

  using KPropsT = KProps;
  using RangeT = Range;
  static constexpr int Dim = syclcompat::detail::range_dimension_v<Range>;

  launch_policy() = delete;
  // Ctor taking a sycl::range<Dim> or sycl::nd_range<Dim>
  launch_policy(Range range, KProps kprops, LProps lprops, size_t lmem_size)
      : range{range}, kernel_properties{kprops}, launch_properties{lprops},
        local_mem_size{lmem_size} {}

  // Ctor taking just dim3 global range (converts to sycl::range<3>)
  launch_policy(dim3 global_range, KProps kprops, LProps lprops,
                  size_t lmem_size)
      : range{global_range}, kernel_properties{kprops},
        launch_properties{lprops}, local_mem_size{lmem_size} {}

  // Ctor taking pair of dim3 (converts to sycl::nd_range<3>)
  launch_policy(dim3 global_range, dim3 work_group_range, KProps kprops,
                  LProps lprops, size_t lmem_size)
      : range{global_range, work_group_range}, kernel_properties{kprops},
        launch_properties{lprops}, local_mem_size{lmem_size} {}

  Range range;
  KProps kernel_properties;
  LProps launch_properties;
  size_t local_mem_size;
};

// Deduction guides for launch_policy dim3 ctors
template <typename KProps, typename LProps>
launch_policy(dim3 global_range, KProps kprops, LProps lprops,
                size_t lmem_size)
    -> launch_policy<sycl::range<3>, KProps, LProps>;

template <typename KProps, typename LProps>
launch_policy(dim3 global_range, dim3 work_group_range, KProps kprops,
                LProps lprops, size_t lmem_size)
    -> launch_policy<sycl::nd_range<3>, KProps, LProps>;

template <typename T> struct is_launch_policy : std::false_type {};

template <typename RangeT, typename KProps, typename LProps>
struct is_launch_policy<launch_policy<RangeT, KProps, LProps>> : std::true_type {};

template <typename T>
inline constexpr bool is_launch_policy_v = is_launch_policy<T>::value;


namespace detail {

template <auto F, typename Range, typename KProps, typename... Args> struct KernelFunctor {
  KernelFunctor(KProps kernel_props, Args... args)
      : kernel_properties{kernel_props},
        argument_tuple(std::make_tuple(args...)) {}

  KernelFunctor(KProps kernel_props, sycl::local_accessor<char, 1> local_acc,
                Args... args)
      : kernel_properties{kernel_props}, local_acc{local_acc},
        argument_tuple(std::make_tuple(args...)) {}

  auto get(sycl_exp::properties_tag) { return kernel_properties; }

  __syclcompat_inline__ inline void
  operator()(syclcompat::detail::range_to_item_t<Range> it) const {
    if constexpr (syclcompat::lmem_invocable<F, Args...>) {
      char *local_mem_ptr = static_cast<char *>(
          local_acc.get_multi_ptr<sycl::access::decorated::no>());
      std::apply(
          [lmem_ptr = local_mem_ptr](auto &&...args) { F(args..., lmem_ptr); },
          argument_tuple);
    } else {
      std::apply([](auto &&...args) { F(args...); }, argument_tuple);
    }
  }

  KProps kernel_properties;
  std::tuple<Args...> argument_tuple;
  sycl::local_accessor<char, 1> local_acc;
};

//====================================================================
// This helper function avoids 2 nested `if constexpr` in detail::launch
template <auto F, typename LaunchStrategy, typename... Args>
auto build_kernel_functor(sycl::handler& cgh, LaunchStrategy launch_policy,
                          Args... args)
    -> KernelFunctor<F, typename LaunchStrategy::RangeT,
                     typename LaunchStrategy::KPropsT, Args...> {
  if constexpr (syclcompat::lmem_invocable<F, Args...>) {
    sycl::local_accessor<char, 1> local_memory(launch_policy.local_mem_size,
                                               cgh);
    return KernelFunctor<F, typename LaunchStrategy::RangeT,
                         typename LaunchStrategy::KPropsT, Args...>(
        launch_policy.kernel_properties, local_memory, args...);
  } else {
      return KernelFunctor<F, typename LaunchStrategy::RangeT,
                        typename LaunchStrategy::KPropsT, Args...>(
              launch_policy.kernel_properties, args...);
  }
}

template <auto F, typename LaunchStrategy, typename... Args>
sycl::event launch(LaunchStrategy launch_policy, sycl::queue q, Args... args) {
  static_assert(syclcompat::args_compatible<F, Args...>);

  sycl_exp::launch_config config(launch_policy.range,
                                 launch_policy.launch_properties);

  return sycl_exp::submit_with_event(q, [&](sycl::handler &cgh) {
    auto KernelFunctor = build_kernel_functor<F>(cgh, launch_policy, args...);
    if constexpr (syclcompat::detail::is_sycl_range<typename LaunchStrategy::RangeT>::value) { //TODO: template aliases for this
      parallel_for(cgh, config, KernelFunctor);
    } else {
      static_assert(
          syclcompat::detail::is_sycl_nd_range<typename LaunchStrategy::RangeT>::value);
      nd_launch(cgh, config, KernelFunctor);
    }
  });
}

} // namespace detail

template <auto F, typename LaunchStrategy, typename... Args>
std::enable_if_t<syclcompat::args_compatible<F, Args...>, sycl::event>
launch(LaunchStrategy launch_policy, sycl::queue q, Args... args) {
  static_assert(is_launch_policy_v<LaunchStrategy>);
  return detail::launch<F>(launch_policy, q, args...);
}


template <auto F, typename LaunchStrategy, typename... Args>
std::enable_if_t<syclcompat::args_compatible<F, Args...>, sycl::event>
launch(LaunchStrategy launch_policy, Args... args) {
  static_assert(is_launch_policy_v<LaunchStrategy>);
  return launch<F>(launch_policy, get_default_queue(), args...);
}


} // namespace experimental
} // namespace syclcompat
