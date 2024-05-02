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
#include <syclcompat/kernel_properties.hpp>
#include <syclcompat/launch.hpp>

#if defined(SYCL_EXT_ONEAPI_KERNEL_PROPERTIES) &&                              \
    defined(SYCL_EXT_ONEAPI_PROPERTIES)
// defined(SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS) uncomment once
// SYCL_EXT_ONEAPI_ENQUEUE_FUNCTIONS is defined

namespace sycl_exp = sycl::ext::oneapi::experimental;

namespace syclcompat {
namespace experimental {
namespace detail {

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

template <auto KernelFunc, typename KernelPropertiesStruct,
          bool UsesLocalMemory, typename... Args>
struct KernelFunctor {
  KernelFunctor(Args... args, char *local_mem_ptr = nullptr)
      : argument_tuple(std::make_tuple(args...)), local_mem_ptr(local_mem_ptr) {
  }

  auto get(sycl_exp::properties_tag) { return kernel_properties; }

  __attribute__((always_inline)) inline void
  operator()(sycl::nd_item<3> it) const {
    if constexpr (UsesLocalMemory) {
      run_kernel<KernelFunc>(
          std::tuple_cat(argument_tuple, std::make_tuple(local_mem_ptr)));
    } else {
      run_kernel<KernelFunc>(argument_tuple);
    }
  }

  std::tuple<Args...> argument_tuple;
  char *local_mem_ptr;
  static constexpr auto kernel_properties =
      KernelPropertiesStruct::kernel_properties;
};
} // namespace detail

template <auto KernelFunc, auto KernelPropertiesStruc, typename PropertyList, typename... Args>
std::enable_if<std::is_invocable_v<decltype(KernelFunc), Args...>, sycl::event>
launch(const sycl::nd_range<3> &launch_params, std::size_t local_memory_size,
       sycl::queue &queue, const PropertyList& launch_properties, Args... args) {
  static_assert(detail::getArgumentCount(KernelFunc) == sizeof...(args) + 1,
                "Wrong number of arguments to SYCL kernel");
  sycl_exp::launch_config config(launch_params, launch_properties);
  return sycl_exp::submit_with_event(queue, [&](sycl::handler &cgh) {
    sycl::local_accessor<char, 1> local_mem(local_memory_size, cgh);
    sycl_exp::nd_launch(
        cgh, config,
        detail::KernelFunctor<KernelFunc, KernelPropertiesStruct, true,
                              Args...>(args..., local_mem.get_pointer()));
  });
}

template <auto KernelFunc, auto KernelPropertiesStruct, typename PropertyList,
          typename... Args>
std::enable_if<std::is_invocable_v<decltype(KernelFunc), Args...>, sycl::event>
launch(const sycl::nd_range<3> &launch_params, sycl::queue &queue,
       const PropertyList &launch_properties, Args... args) {
  static_assert(detail::getArgumentCount(KernelFunc) == sizeof...(args),
                "Wrong number of arguments to SYCL kernel");
  sycl_exp::launch_config config(launch_params, launch_properties);
  return sycl_exp::submit_with_event(queue, [&](sycl::handler &cgh) {
    sycl_exp::nd_launch(
        cgh, config,
        detail::KernelFunctor<KernelFunc, KernelPropertiesStruct, false,
                              Args...>(args...));
  });
}

//==================================================================================//

template <auto KernelFunc, auto KernelPropertiesStruct, typename PropertyList, typename... Args>
std::enable_if<std::is_invocable_v<decltype(KernelFunc), Args...>, sycl::event>
launch(const sycl::range<3> &global_range, const sycl::range<3> &local_range,
       sycl::queue &queue, const PropertyList& launch_properties, Args... args) {
  launch<KernelFunc, KernelPropertiesStruct>(
      sycl::nd_range<3>(global_range, local_range), queue, launch_properties, args...);
}

template <auto KernelFunc, auto KernelPropertiesStruct, typename PropertyList, typename... Args>
std::enable_if<std::is_invocable_v<decltype(KernelFunc), Args...>, sycl::event>
launch(const sycl::range<3> &global_range, const sycl::range<3> &local_range,
      const PropertyList& launch_properties,  Args... args) {
  launch<KernelFunc, KernelPropertiesStruct>(
      global_range, local_range, get_default_queue(), launch_properties, args...);
}

template <auto KernelFunc, auto KernelPropertiesStruct, typename PropertyList, typename... Args>
std::enable_if<std::is_invocable_v<decltype(KernelFunc), Args...>, sycl::event>
launch(const dim3 &grid_dim, const dim3 &block_dim, sycl::queue &queue,
      const PropertyList& launch_properties, Args... args) {
  launch<KernelFunc, KernelPropertiesStruct>(
      sycl::nd_range<3>(grid_dim * block_dim, block_dim), queue, launch_properties, args...);
}

template <auto KernelFunc, auto KernelPropertiesStruct, typename PropertyList, typename... Args>
std::enable_if<std::is_invocable_v<decltype(KernelFunc), Args...>, sycl::event>
launch(const dim3 &grid_dim, const dim3 &block_dim, const PropertyList& launch_properties, Args... args) {
  launch<KernelFunc, KernelPropertiesStruct>(
      grid_dim, block_dim, get_default_queue(), launch_properties, args...);
}

template <auto KernelFunc, auto KernelPropertiesStruct, typename... Args>
std::enable_if<std::is_invocable_v<decltype(KernelFunc), Args...>, sycl::event>
launch(const sycl::range<3> &global_range, const sycl::range<3> &local_range, Args... args) {
  launch<KernelFunc, KernelPropertiesStruct>(
      global_range, local_range, get_default_queue(), empty_property_list, args...);
}

template <auto KernelFunc, typename LaunchProperties, typename... Args>
std::enable_if<std::is_invocable_v<decltype(KernelFunc), Args...>, sycl::event>
launch(const sycl::range<3> &global_range, const sycl::range<3> &local_range, const LaunchProperties& launch_properties, Args... args) {
  launch<KernelFunc, EmptyKernelPropertyStruct>(
      global_range, local_range, get_default_queue(), launch_properties, args...);
}

template <auto KernelFunc, auto KernelPropertiesStruct, typename... Args>
std::enable_if<std::is_invocable_v<decltype(KernelFunc), Args...>, sycl::event>
launch(const dim3 &grid_dim, const dim3 &block_dim, Args... args) {
  launch<KernelFunc, KernelPropertiesStruct>(
      grid_dim, block_dim, get_default_queue(), empty_property_list, args...);
}

template <auto KernelFunc, typename PropertyList, typename... Args>
std::enable_if<std::is_invocable_v<decltype(KernelFunc), Args...>, sycl::event>
launch(const dim3 &grid_dim, const dim3 &block_dim, const PropertyList& launch_properties, Args... args) {
  launch<KernelFunc, EmptyKernelPropertyStruct>(
      grid_dim, block_dim, get_default_queue(), launch_properties, args...);
}


///==============================================================================================///

template <auto KernelFunc, auto KernelPropertiesStruct, typename PropertyList, typename... Args>
std::enable_if<std::is_invocable_v<decltype(KernelFunc), Args...>, sycl::event>
launch(const sycl::range<3> &global_range, const sycl::range<3> &local_range,
       std::size_t local_mem_size, sycl::queue &queue, const PropertyList& launch_properties, Args... args) {
  launch<KernelFunc, KernelPropertiesStruct>(
      sycl::nd_range<3>(global_range, local_range), local_mem_size, queue,
      launch_properties, args...);
}

template <auto KernelFunc, auto KernelPropertiesStruct, typename PropertyList, typename... Args>
std::enable_if<std::is_invocable_v<decltype(KernelFunc), Args...>, sycl::event>
launch(const sycl::range<3> &global_range, const sycl::range<3> &local_range,
       std::size_t local_mem_size, const PropertyList& launch_properties, Args... args) {
  launch<KernelFunc, KernelPropertiesStruct>(
      global_range, local_range, local_mem_size, get_default_queue(), launch_properties, args...);
}

template <auto KernelFunc, auto KernelPropertiesStruct, typename PropertyList, typename... Args>
std::enable_if<std::is_invocable_v<decltype(KernelFunc), Args...>, sycl::event>
launch(const dim3 &grid_dim, const dim3 &block_dim, std::size_t local_mem_size,
       sycl::queue &queue, const PropertyList& launch_properties, Args... args) {
  launch<KernelFunc, KernelPropertiesStruct>(
      sycl::nd_range<3>(grid_dim * block_dim, block_dim), local_mem_size, queue,
      launch_properties, args...);
}

template <auto KernelFunc, auto KernelPropertiesStruct, typename PropertyList, typename... Args>
std::enable_if<std::is_invocable_v<decltype(KernelFunc), Args...>, sycl::event>
launch(const dim3 &grid_dim, const dim3 &block_dim, std::size_t local_mem_size,
       const PropertyList& launch_properties, Args... args) {
  launch<KernelFunc, KernelPropertiesStruct>(
      grid_dim, block_dim, local_mem_size, get_default_queue(), launch_properties, args...);
}

template <auto KernelFunc, auto KernelPropertiesStruct, typename... Args>
std::enable_if<std::is_invocable_v<decltype(KernelFunc), Args...>, sycl::event>
launch(const dim3 &grid_dim, const dim3 &block_dim, std::size_t local_mem_size,
       Args... args) {
  launch<KernelFunc, KernelPropertiesStruct>(
      grid_dim, block_dim, local_mem_size, get_default_queue(), empty_property_list, args...);
}

template <auto KernelFunc, typename PropertyList, typename... Args>
std::enable_if<std::is_invocable_v<decltype(KernelFunc), Args...>, sycl::event>
launch(const dim3 &grid_dim, const dim3 &block_dim, std::size_t local_mem_size, const PropertyList& launch_properties,
       Args... args) {
  launch<KernelFunc, EmptyKernelPropertyStruct>(
      grid_dim, block_dim, local_mem_size, get_default_queue(), launch_properties, args...);
}

template <auto KernelFunc, auto KernelPropertiesStruct, typename... Args>
std::enable_if<std::is_invocable_v<decltype(KernelFunc), Args...>, sycl::event>
launch(const sycl::range<3> &global_range, const sycl::range<3> &local_range, std::size_t local_mem_size,
       Args... args) {
  launch<KernelFunc, KernelPropertiesStruct>(
      global_range, local_range, local_mem_size, get_default_queue(), empty_property_list, args...);
}

template <auto KernelFunc, typename PropertyList, typename... Args>
std::enable_if<std::is_invocable_v<decltype(KernelFunc), Args...>, sycl::event>
launch(const sycl::range<3> &global_range, const sycl::range<3> &local_range, std::size_t local_mem_size, const PropertyList& launch_properties,
       Args... args) {
  launch<KernelFunc, EmptyKernelPropertyStruct>(
      global_range, local_range, local_mem_size, get_default_queue(), launch_properties, args...);
}

} // namespace experimental
} // namespace syclcompat

#endif
