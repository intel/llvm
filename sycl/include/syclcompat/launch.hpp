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

#include <sycl/accessor.hpp>
#include <sycl/event.hpp>
#include <sycl/nd_range.hpp>
#include <sycl/queue.hpp>
#include <sycl/range.hpp>
#include <sycl/reduction.hpp>

#include <syclcompat/device.hpp>
#include <syclcompat/dims.hpp>

namespace syclcompat {

namespace detail {

template <typename R, typename... Types>
constexpr size_t getArgumentCount(R (*f)(Types...)) {
  return sizeof...(Types);
}

template <int Dim>
sycl::nd_range<3> transform_nd_range(const sycl::nd_range<Dim> &range) {
  sycl::range<Dim> global_range = range.get_global_range();
  sycl::range<Dim> local_range = range.get_local_range();
  if constexpr (Dim == 3) {
    return range;
  } else if constexpr (Dim == 2) {
    return sycl::nd_range<3>{{1, global_range[0], global_range[1]},
                             {1, local_range[0], local_range[1]}};
  }
  return sycl::nd_range<3>{{1, 1, global_range[0]}, {1, 1, local_range[0]}};
}

template <auto F, typename... Args>
std::enable_if_t<std::is_invocable_v<decltype(F), Args...>, sycl::event>
launch(const sycl::nd_range<3> &range, sycl::queue q, Args... args) {
  static_assert(detail::getArgumentCount(F) == sizeof...(args),
                "Wrong number of arguments to SYCL kernel");
  static_assert(
      std::is_same<std::invoke_result_t<decltype(F), Args...>, void>::value,
      "SYCL kernels should return void");

  return q.parallel_for(
      range, [=](sycl::nd_item<3>) { [[clang::always_inline]] F(args...); });
}

template <auto F, typename... Args>
sycl::event launch(const sycl::nd_range<3> &range, size_t mem_size,
                   sycl::queue q, Args... args) {
  static_assert(detail::getArgumentCount(F) == sizeof...(args) + 1,
                "Wrong number of arguments to SYCL kernel");

  using F_t = decltype(F);
  using f_return_t = typename std::invoke_result_t<F_t, Args..., char *>;
  static_assert(std::is_same<f_return_t, void>::value,
                "SYCL kernels should return void");

  return q.submit([&](sycl::handler &cgh) {
    auto local_acc = sycl::local_accessor<char, 1>(mem_size, cgh);
    cgh.parallel_for(range, [=](sycl::nd_item<3>) {
      auto local_mem = local_acc.get_pointer();
      [[clang::always_inline]] F(args..., local_mem);
    });
  });
}

} // namespace detail

template <int Dim>
sycl::nd_range<Dim> compute_nd_range(sycl::range<Dim> global_size_in,
                                     sycl::range<Dim> work_group_size) {

  if (global_size_in.size() == 0 || work_group_size.size() == 0) {
    throw std::invalid_argument("Global or local size is zero!");
  }
  for (size_t i = 0; i < Dim; ++i) {
    if (global_size_in[i] < work_group_size[i])
      throw std::invalid_argument("Work group size larger than global size");
  }

  auto global_size =
      ((global_size_in + work_group_size - 1) / work_group_size) *
      work_group_size;
  return {global_size, work_group_size};
}

inline sycl::nd_range<1> compute_nd_range(int global_size_in,
                                          int work_group_size) {
  return compute_nd_range<1>(global_size_in, work_group_size);
}

template <auto F, int Dim, typename... Args>
std::enable_if_t<std::is_invocable_v<decltype(F), Args...>, sycl::event>
launch(const sycl::nd_range<Dim> &range, sycl::queue q, Args... args) {
  return detail::launch<F>(detail::transform_nd_range<Dim>(range), q, args...);
}

template <auto F, int Dim, typename... Args>
std::enable_if_t<std::is_invocable_v<decltype(F), Args...>, sycl::event>
launch(const sycl::nd_range<Dim> &range, Args... args) {
  return launch<F>(range, get_default_queue(), args...);
}

// Alternative launch through dim3 objects
template <auto F, typename... Args>
std::enable_if_t<std::is_invocable_v<decltype(F), Args...>, sycl::event>
launch(const dim3 &grid, const dim3 &threads, sycl::queue q, Args... args) {
  return launch<F>(sycl::nd_range<3>{grid * threads, threads}, q, args...);
}

template <auto F, typename... Args>
std::enable_if_t<std::is_invocable_v<decltype(F), Args...>, sycl::event>
launch(const dim3 &grid, const dim3 &threads, Args... args) {
  return launch<F>(grid, threads, get_default_queue(), args...);
}

/// Launches a kernel with the templated F param and arguments on a
/// device specified by the given nd_range and SYCL queue.
/// @tparam F SYCL kernel to be executed, expects signature F(T* local_mem,
/// Args... args).
/// @tparam Dim nd_range dimension number.
/// @tparam Args Types of the arguments to be passed to the kernel.
/// @param range Nd_range specifying the work group and global sizes for the
/// kernel.
/// @param q The SYCL queue on which to execute the kernel.
/// @param mem_size The size, in number of bytes, of the local
/// memory to be allocated for kernel.
/// @param args The arguments to be passed to the kernel.
/// @return A SYCL event object that can be used to synchronize with the
/// kernel's execution.
template <auto F, int Dim, typename... Args>
sycl::event launch(const sycl::nd_range<Dim> &range, size_t mem_size,
                   sycl::queue q, Args... args) {
  return detail::launch<F>(detail::transform_nd_range<Dim>(range), mem_size, q,
                           args...);
}

/// Launches a kernel with the templated F param and arguments on a
/// device specified by the given nd_range using theSYCL default queue.
/// @tparam F SYCL kernel to be executed, expects signature F(T* local_mem,
/// Args... args).
/// @tparam Dim nd_range dimension number.
/// @tparam Args Types of the arguments to be passed to the kernel.
/// @param range Nd_range specifying the work group and global sizes for the
/// kernel.
/// @param mem_size The size, in number of bytes, of the local
/// memory to be allocated for kernel.
/// @param args The arguments to be passed to the kernel.
/// @return A SYCL event object that can be used to synchronize with the
/// kernel's execution.
template <auto F, int Dim, typename... Args>
sycl::event launch(const sycl::nd_range<Dim> &range, size_t mem_size,
                   Args... args) {
  return launch<F>(range, mem_size, get_default_queue(), args...);
}

/// Launches a kernel with the templated F param and arguments on a
/// device with a user-specified grid and block dimensions following the
/// standard of other programming models using a user-defined SYCL queue.
/// @tparam F SYCL kernel to be executed, expects signature F(T* local_mem,
/// Args... args).
/// @tparam Dim nd_range dimension number.
/// @tparam Args Types of the arguments to be passed to the kernel.
/// @param grid Grid dimensions represented with an (x, y, z) iteration space.
/// @param threads Block dimensions represented with an (x, y, z) iteration
/// space.
/// @param mem_size The size, in number of bytes, of the local
/// memory to be allocated for kernel.
/// @param args The arguments to be passed to the kernel.
/// @return A SYCL event object that can be used to synchronize with the
/// kernel's execution.
template <auto F, typename... Args>
sycl::event launch(const dim3 &grid, const dim3 &threads, size_t mem_size,
                   sycl::queue q, Args... args) {
  return launch<F>(sycl::nd_range<3>{grid * threads, threads}, mem_size, q,
                   args...);
}

/// Launches a kernel with the templated F param and arguments on a
/// device with a user-specified grid and block dimensions following the
/// standard of other programming models using the default SYCL queue.
/// @tparam F SYCL kernel to be executed, expects signature F(T* local_mem,
/// Args... args).
/// @tparam Dim nd_range dimension number.
/// @tparam Args Types of the arguments to be passed to the kernel.
/// @param grid Grid dimensions represented with an (x, y, z) iteration space.
/// @param threads Block dimensions represented with an (x, y, z) iteration
/// space.
/// @param mem_size The size, in number of bytes, of the
/// local memory to be allocated.
/// @param args The arguments to be passed to the kernel.
/// @return A SYCL event object that can be used to synchronize with the
/// kernel's execution.
template <auto F, typename... Args>
sycl::event launch(const dim3 &grid, const dim3 &threads, size_t mem_size,
                   Args... args) {
  return launch<F>(grid, threads, mem_size, get_default_queue(), args...);
}

} // namespace syclcompat
