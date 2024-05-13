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
 *  SYCLcompat
 *
 *  launch_experimental.hpp
 *
 *  Description:
 *    Provides the interface for launching a kernel with launch properties as
 *    well as kernel properties, using the enqueue_functions extensions for
 *    launching the kernel.
 *    Depends on the following 3 extensions-
 *       sycl_ext_oneapi_kernel_properties
 *       sycl_ext_oneapi_enqueue_functions
 *       sycl_ext_oneapi_properties
 **************************************************************************/

#pragma once

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>

#include <sycl/sycl.hpp>

#include <tuple>

#include <syclcompat/device.hpp>
#include <syclcompat/dims.hpp>
#include <syclcompat/launch.hpp>

#if defined(SYCL_EXT_ONEAPI_KERNEL_PROPERTIES) &&                              \
    defined(SYCL_EXT_ONEAPI_PROPERTIES)
// defined(SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS) uncomment once
// SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS is defined

namespace sycl_exp = sycl::ext::oneapi::experimental;

namespace syclcompat {
namespace experimental {

namespace detail {
constexpr auto empty_property_list =
    sycl::ext::oneapi::experimental::properties{};

template <typename T> struct is_property_list_type : std::false_type {};

template <typename T>
struct is_property_list_type<sycl_exp::properties<T>> : std::true_type {};
} // namespace detail

//================================================================================================//
// Overloads using Local Memory //
//================================================================================================//

template <typename KernelFunctor, typename PropertyList, typename... Args>
inline std::enable_if_t<std::is_invocable_v<KernelFunctor, sycl::nd_item<3>>,
                        sycl::event>
launch(const sycl::nd_range<3> &launch_params, std::size_t local_memory_size,
       const PropertyList &launch_properties, const sycl::queue &queue,
       Args... args) {
  sycl_exp::launch_config config(launch_params, launch_properties);
  return sycl_exp::submit_with_event(queue, [&](sycl::handler &cgh) {
    sycl::local_accessor<char, 1> local_memory(local_memory_size, cgh);
    sycl_exp::nd_launch(cgh, config,
                        KernelFunctor(args..., local_memory.get_multi_ptr<sycl::access::decorated::true>()));
  });
}

template <typename KernelFunctor, int Dim, typename PropertyList,
          typename... Args>
inline std::enable_if_t<std::is_invocable_v<KernelFunctor, sycl::nd_item<3>> &&
                            detail::is_property_list_type<PropertyList>::value,
                        sycl::event>
launch(const sycl::nd_range<Dim> &launch_params, std::size_t local_memory_size,
       const PropertyList &launch_properties, Args... args) {
  return launch<KernelFunctor>(
      ::syclcompat::detail::transform_nd_range(launch_params),
      local_memory_size, launch_properties, ::syclcompat::get_default_queue(),
      args...);
}

template <typename KernelFunctor, int Dim, typename... Args>
inline std::enable_if_t<std::is_invocable_v<KernelFunctor, sycl::nd_item<3>>,
                        sycl::event>
launch(const sycl::nd_range<Dim> &launch_params, std::size_t local_memory_size,
       Args... args) {
  using PropertyList = decltype(detail::empty_property_list);
  return launch<KernelFunctor>(
      ::syclcompat::detail::transform_nd_range(launch_params),
      local_memory_size, detail::empty_property_list, args...);
}

template <typename KernelFunctor, int Dim, typename PropertyList,
          typename... Args>
inline std::enable_if_t<std::is_invocable_v<KernelFunctor, sycl::nd_item<3>> &&
                            detail::is_property_list_type<PropertyList>::value,
                        sycl::event>
launch(const sycl::range<Dim> &global_range,
       const sycl::range<Dim> &local_range, std::size_t local_memory_size,
       const PropertyList &launch_properties, Args... args) {
  return launch<KernelFunctor>(
      ::syclcompat::detail::transform_nd_range(
          sycl::nd_range<Dim>(global_range, local_range)),
      local_memory_size, launch_properties, ::syclcompat::get_default_queue(),
      args...);
}

template <typename KernelFunctor, int Dim, typename... Args>
inline std::enable_if_t<std::is_invocable_v<KernelFunctor, sycl::nd_item<3>>,
                        sycl::event>
launch(const sycl::range<Dim> &global_range,
       const sycl::range<Dim> &local_range, std::size_t local_memory_size,
       Args... args) {
  using PropertyList = decltype(detail::empty_property_list);
  return launch<KernelFunctor>(
      ::syclcompat::detail::transform_nd_range(
          sycl::nd_range<Dim>(global_range, local_range)),
      local_memory_size, detail::empty_property_list, args...);
}

template <typename KernelFunctor, typename PropertyList, typename... Args>
inline std::enable_if_t<std::is_invocable_v<KernelFunctor, sycl::nd_item<3>> &&
                            detail::is_property_list_type<PropertyList>::value,
                        sycl::event>
launch(const dim3 &grid_dim, const dim3 &block_dim,
       std::size_t local_memory_size, const PropertyList &launch_properties,
       Args... args) {
  return launch<KernelFunctor>(sycl::nd_range<3>(grid_dim * block_dim, block_dim),
                               local_memory_size, launch_properties,
                               ::syclcompat::get_default_queue(), args...);
}

template <typename KernelFunctor, typename... Args>
inline std::enable_if_t<std::is_invocable_v<KernelFunctor, sycl::nd_item<3>>,
                        sycl::event>
launch(const dim3 &grid_dim, const dim3 &block_dim,
       std::size_t local_memory_size, Args... args) {
  using PropertyList = decltype(detail::empty_property_list);
  return launch<KernelFunctor>(
      sycl::nd_range<3>(grid_dim * block_dim, block_dim), local_memory_size,
      detail::empty_property_list, args...);
}

//================================================================================================//
// Overloads not using Local Memory //
//================================================================================================//

template <typename KernelFunctor, typename PropertyList, typename... Args>
inline std::enable_if_t<std::is_invocable_v<KernelFunctor, sycl::nd_item<3>>,
                        sycl::event>
launch(sycl::nd_range<3> launch_params, const PropertyList &launch_properties,
       const sycl::queue &queue, Args... args) {
  sycl_exp::launch_config config(launch_params, launch_properties);
  return sycl_exp::submit_with_event(queue, [&](sycl::handler &cgh) {
    sycl_exp::nd_launch(cgh, config, KernelFunctor(args...));
  });
}

template <typename KernelFunctor, int Dim, typename PropertyList,
          typename... Args>
inline std::enable_if_t<std::is_invocable_v<KernelFunctor, sycl::nd_item<3>> &&
                            detail::is_property_list_type<PropertyList>::value,
                        sycl::event>
launch(sycl::nd_range<Dim> launch_params, const PropertyList &launch_properties,
       Args... args) {
  return launch<KernelFunctor>(
      ::syclcompat::detail::transform_nd_range(launch_params),
      launch_properties, ::syclcompat::get_default_queue(), args...);
}

template <typename KernelFunctor, int Dim, typename... Args>
inline std::enable_if_t<std::is_invocable_v<KernelFunctor, sycl::nd_item<3>>,
                        sycl::event>
launch(sycl::nd_range<Dim> launch_params, Args... args) {
  using PropertyList = decltype(detail::empty_property_list);
  return launch<KernelFunctor>(
      ::syclcompat::detail::transform_nd_range(launch_params),
      detail::empty_property_list, args...);
}

template <typename KernelFunctor, int Dim, typename PropertyList,
          typename... Args>
inline std::enable_if_t<std::is_invocable_v<KernelFunctor, sycl::nd_item<3>> &&
                            detail::is_property_list_type<PropertyList>::value,
                        sycl::event>
launch(sycl::range<Dim> global_range, sycl::range<Dim> local_range,
       const PropertyList &launch_properties, Args... args) {
  return launch<KernelFunctor>(
      ::syclcompat::detail::transform_nd_range(
          sycl::nd_range<3>(global_range, local_range)),
      launch_properties, ::syclcompat::get_default_queue(), args...);
}

template <typename KernelFunctor, int Dim, typename... Args>
inline std::enable_if_t<std::is_invocable_v<KernelFunctor, sycl::nd_item<3>>,
                        sycl::event>
launch(sycl::range<Dim> global_range, sycl::range<Dim> local_range,
       Args... args) {
  using PropertyList = decltype(detail::empty_property_list);
  return launch<KernelFunctor>(
      ::syclcompat::detail::transform_nd_range(
          sycl::nd_range<Dim>(global_range, local_range)),
      detail::empty_property_list, args...);
}

template <typename KernelFunctor, typename PropertyList, typename... Args>
inline std::enable_if_t<std::is_invocable_v<KernelFunctor, sycl::nd_item<3>> &&
                            detail::is_property_list_type<PropertyList>::value,
                        sycl::event>
launch(const dim3 &grid_dim, const dim3 &block_dim,
       const PropertyList &launch_properties, Args... args) {
  return launch<KernelFunctor>(sycl::nd_range<3>(grid_dim * block_dim, block_dim),
                               launch_properties,
                               ::syclcompat::get_default_queue(), args...);
}

template <typename KernelFunctor, typename... Args>
inline std::enable_if_t<std::is_invocable_v<KernelFunctor, sycl::nd_item<3>>,
                        sycl::event>
launch(const dim3 &grid_dim, const dim3 &block_dim, Args... args) {
  using PropertyList = decltype(detail::empty_property_list);
  return launch<KernelFunctor>(
      sycl::nd_range<3>(grid_dim * block_dim, block_dim), detail::empty_property_list,
      args...);
}

} // namespace experimental
} // namespace syclcompat

#endif
