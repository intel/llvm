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
 *  Launch Overloads with accepting required subgroup size
 **************************************************************************/

#pragma once

#include <syclcompat/device.hpp>
#include <syclcompat/dims.hpp>
#include <syclcompat/launch.hpp>

namespace syclcompat {
namespace experimental {

//================================================================================================//
// Overloads using Local Memory //
//================================================================================================//

template <auto F, int SubgroupSize, typename... Args>
std::enable_if_t<std::is_invocable_v<decltype(F), Args..., char *>, sycl::event>
launch(sycl::nd_range<3> launch_range, std::size_t local_memory_size,
       sycl::queue queue, Args... args) {
  return queue.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<char, 1> loc(local_memory_size, cgh);
    cgh.parallel_for(
        launch_range,
        [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(SubgroupSize)]] {
          [[clang::always_inline]] F(
              args..., loc.get_multi_ptr<sycl::access::decorated::yes>());
        });
  });
}

template <auto F, int SubgroupSize, int Dim, typename... Args>
std::enable_if_t<std::is_invocable_v<decltype(F), Args..., char *>, sycl::event>
launch(sycl::nd_range<Dim> launch_range, std::size_t local_memory_size,
       Args... args) {
  return launch<F, SubgroupSize, Args...>(
      ::syclcompat::detail::transform_nd_range(launch_range), local_memory_size,
      ::syclcompat::get_default_queue(), args...);
}

template <auto F, int SubgroupSize, typename... Args>
std::enable_if_t<std::is_invocable_v<decltype(F), Args..., char *>, sycl::event>
launch(::syclcompat::dim3 grid_dim, ::syclcompat::dim3 block_dim,
       std::size_t local_memory_size, Args... args) {
  return launch<F, SubgroupSize, Args...>(
      ::syclcompat::detail::transform_nd_range(sycl::nd_range(
          sycl::range<3>(grid_dim * block_dim), sycl::range<3>(block_dim))),
      local_memory_size, ::syclcompat::get_default_queue(), args...);
}

//================================================================================================//
// Overloads not using Local Memory //
//================================================================================================//

template <auto F, int SubgroupSize, typename... Args>
std::enable_if_t<std::is_invocable_v<decltype(F), Args...>, sycl::event>
launch(sycl::nd_range<3> launch_range, sycl::queue queue, Args... args) {
  return queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(launch_range,
                     [=](sycl::nd_item<3> it)
                         [[sycl::reqd_sub_group_size(SubgroupSize)]] {
                           [[clang::always_inline]] F(args...);
                         });
  });
}

template <auto F, int SubgroupSize, int Dim, typename... Args>
std::enable_if_t<std::is_invocable_v<decltype(F), Args...>, sycl::event>
launch(sycl::nd_range<Dim> launch_range, Args... args) {
  return launch<F, SubgroupSize, Args...>(
      ::syclcompat::detail::transform_nd_range(launch_range),
      ::syclcompat::get_default_queue(), args...);
}

template <auto F, int SubgroupSize, typename... Args>
std::enable_if_t<std::is_invocable_v<decltype(F), Args...>, sycl::event>
launch(::syclcompat::dim3 grid_dim, ::syclcompat::dim3 block_dim,
       Args... args) {
  return launch<F, SubgroupSize, Args...>(
      ::syclcompat::detail::transform_nd_range(sycl::nd_range(
          sycl::range<3>(grid_dim * block_dim), sycl::range<3>(block_dim))),
      ::syclcompat::get_default_queue(), args...);
}

} // namespace experimental
} // namespace syclcompat
