//==--- user_defined_reductions.hpp -- SYCL ext header file -=--*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/builtins.hpp> // for sycl::min
#include <sycl/detail/defines.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/sycl_span.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {
template <typename GroupHelper, typename T, typename BinaryOperation>
T reduce_over_group_impl(GroupHelper group_helper, T x, size_t num_elements,
                         BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  auto g = group_helper.get_group();
  // IMPORTANT: num_elements is *guaranteed* to be less than
  // g.get_local_linear_id()!

  // It seems shift_group_left is overly restrictive and requires trivial types
  // instead of trivially copyable.
  if constexpr (sycl::detail::is_sub_group<decltype(g)>::value &&
                std::is_trivial_v<T>) {
    // sycl::ext::oneapi::sub_group isn't sycl::sub_group, and shift_group_left
    // only accepts the latter.
    auto ndi = sycl::ext::oneapi::experimental::this_nd_item<
        decltype(g)::dimensions>();
    auto sg = ndi.get_sub_group();
    for (size_t offset = num_elements / 2; offset > 0; offset /= 2) {
      auto y = shift_group_left(sg, x, offset);
      // y is unspecified for the work items with id higher than offset. In
      // theory, it might have a value that would cause unwanted side effects in
      // binary_op, so only apply the operation in the work items that are using
      // "live" values.
      if (sg.get_local_linear_id() < offset)
        x = binary_op(x, y);
    }
    return group_broadcast(sg, x);
  } else {
    T *Memory = reinterpret_cast<T *>(group_helper.get_memory().data());
    if constexpr (sycl::detail::is_group<decltype(g)>::value &&
                  std::is_trivial_v<T>) {
      // TODO: Use get_child_group from sycl_ext_oneapi_root_group extension
      // once it is implemented instead of this free function.
      auto ndi = sycl::ext::oneapi::experimental::this_nd_item<
          decltype(g)::dimensions>();
      auto sg = ndi.get_sub_group();

      auto lid = g.get_local_linear_id();
      auto sg_lid = sg.get_local_linear_id();
      auto sg_max_size = sg.get_max_local_range()[0];
      auto sg_size = sg.get_local_linear_range();

      auto sg_leader_lid = group_broadcast(sg, lid);

      do {
        if (num_elements <= 1) {
          return group_broadcast(g, x);
        }

        if (num_elements % sg_max_size != 0) {
          // The way work group is split into sub-groups is implementation
          // defined and handling it in generic way would complicate things.
          break;
        }

        auto sg_num_elements =
            sg_leader_lid < num_elements
                ? sycl::min<size_t>(sg_size, num_elements - sg_leader_lid)
                : 0;
        group_barrier(g);
        if (sg_num_elements > 0) {
          auto reduce_sg = reduce_over_group_impl(
              group_with_scratchpad<sycl::sub_group, 0>{
                  sg, sycl::span<std::byte, 0>{}},
              x, sg_num_elements, binary_op);
          Memory[sg.get_group_linear_id()] = reduce_sg;
        }
        group_barrier(g);
        num_elements = num_elements / sg_max_size;
        if (lid < num_elements)
          x = Memory[lid];
      } while (true);
    }

    Memory[g.get_local_linear_id()] = x;
    group_barrier(g);
    T result = Memory[0];
    if (g.leader()) {
      for (int i = 1; i < num_elements; i++) {
        result = binary_op(result, Memory[i]);
      }
    }
    group_barrier(g);
    return group_broadcast(g, result);
  }
#else
  std::ignore = group_helper;
  std::ignore = x;
  std::ignore = num_elements;
  std::ignore = binary_op;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}
} // namespace detail

// ---- reduce_over_group
template <typename GroupHelper, typename T, typename BinaryOperation>
std::enable_if_t<(is_group_helper_v<GroupHelper>), T>
reduce_over_group(GroupHelper group_helper, T x, BinaryOperation binary_op) {
  if constexpr (sycl::detail::is_native_op<T, BinaryOperation>::value) {
    return sycl::reduce_over_group(group_helper.get_group(), x, binary_op);
  }
#ifdef __SYCL_DEVICE_ONLY__
  return detail::reduce_over_group_impl(
      group_helper, x, group_helper.get_group().get_local_linear_range(),
      binary_op);
#else
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename GroupHelper, typename V, typename T,
          typename BinaryOperation>
std::enable_if_t<(is_group_helper_v<GroupHelper>), T>
reduce_over_group(GroupHelper group_helper, V x, T init,
                  BinaryOperation binary_op) {
  if constexpr (sycl::detail::is_native_op<V, BinaryOperation>::value &&
                sycl::detail::is_native_op<T, BinaryOperation>::value) {
    return sycl::reduce_over_group(group_helper.get_group(), x, init,
                                   binary_op);
  }
#ifdef __SYCL_DEVICE_ONLY__
  return binary_op(init, reduce_over_group(group_helper, x, binary_op));
#else
  std::ignore = group_helper;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

// ---- joint_reduce
template <typename GroupHelper, typename Ptr, typename BinaryOperation>
std::enable_if_t<(is_group_helper_v<GroupHelper> &&
                  sycl::detail::is_pointer_v<Ptr>),
                 typename std::iterator_traits<Ptr>::value_type>
joint_reduce(GroupHelper group_helper, Ptr first, Ptr last,
             BinaryOperation binary_op) {
  if constexpr (sycl::detail::is_native_op<
                    typename std::iterator_traits<Ptr>::value_type,
                    BinaryOperation>::value) {
    return sycl::joint_reduce(group_helper.get_group(), first, last, binary_op);
  }
#ifdef __SYCL_DEVICE_ONLY__
  // TODO: the complexity is linear and not logarithmic. Something like
  // https://github.com/intel/llvm/blob/8ebd912679f27943d8ef6c33a9775347dce6b80d/sycl/include/sycl/reduction.hpp#L1810-L1818
  // might be applicable here.
  using T = typename std::iterator_traits<Ptr>::value_type;
  auto g = group_helper.get_group();
  T partial = *(first + g.get_local_linear_id());
  Ptr second = first + g.get_local_linear_range();
  sycl::detail::for_each(g, second, last,
                         [&](const T &x) { partial = binary_op(partial, x); });
  group_barrier(g);
  size_t num_elements = last - first;
  num_elements = std::min(num_elements, g.get_local_linear_range());
  return detail::reduce_over_group_impl(group_helper, partial, num_elements,
                                        binary_op);
#else
  std::ignore = group_helper;
  std::ignore = first;
  std::ignore = last;
  std::ignore = binary_op;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename GroupHelper, typename Ptr, typename T,
          typename BinaryOperation>
std::enable_if_t<
    (is_group_helper_v<GroupHelper> && sycl::detail::is_pointer_v<Ptr>), T>
joint_reduce(GroupHelper group_helper, Ptr first, Ptr last, T init,
             BinaryOperation binary_op) {
  if constexpr (sycl::detail::is_native_op<T, BinaryOperation>::value) {
    return sycl::joint_reduce(group_helper.get_group(), first, last, init,
                              binary_op);
  }
#ifdef __SYCL_DEVICE_ONLY__
  return binary_op(init, joint_reduce(group_helper, first, last, binary_op));
#else
  std::ignore = group_helper;
  std::ignore = last;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
