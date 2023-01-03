//==--- user_defined_reductions.hpp -- SYCL ext header file -=--*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines.hpp>
#include <sycl/group_algorithm.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi::experimental {

// ---- reduce_over_group
template <typename GroupHelper, typename T, typename BinaryOperation>
std::enable_if_t<(is_group_helper_v<GroupHelper>), T>
reduce_over_group(GroupHelper group_helper, T x, BinaryOperation binary_op) {
  if constexpr (sycl::detail::is_native_op<T, BinaryOperation>::value) {
    return sycl::reduce_over_group(group_helper.get_group(), x, binary_op);
  }
#ifdef __SYCL_DEVICE_ONLY__
  T *Memory = reinterpret_cast<T *>(group_helper.get_memory().data());
  auto g = group_helper.get_group();
  Memory[g.get_local_linear_id()] = x;
  group_barrier(g);
  T result = Memory[0];
  if (g.leader()) {
    for (int i = 1; i < g.get_local_linear_range(); i++) {
      result = binary_op(result, Memory[i]);
    }
  }
  group_barrier(g);
  return group_broadcast(g, result);
#else
  std::ignore = group_helper;
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
                  sycl::detail::is_pointer<Ptr>::value),
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
  return reduce_over_group(group_helper, partial, binary_op);
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
    (is_group_helper_v<GroupHelper> && sycl::detail::is_pointer<Ptr>::value), T>
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
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
