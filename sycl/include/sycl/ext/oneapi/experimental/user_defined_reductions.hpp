//==-------------------- user_defined_reductions.hpp -----------------------==//
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
sycl::detail::enable_if_t<(is_group_helper_v<std::decay_t<GroupHelper>>), T>
reduce_over_group(GroupHelper g, T x, BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  T *Memory = reinterpret_cast<T *>(g.get_memory().data());
  Memory[g.get_group().get_local_linear_id()] = x;
  group_barrier(g.get_group());
  T result;
  if (g.get_group().leader()) {
    for (int i = 0; i < g.get_group().get_local_linear_range(); i++) {
      result = binary_op(result, Memory[i]);
    }
  }
  group_barrier(g.get_group());
  return result;
#else
  (void)g;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename GroupHelper, typename V, typename T,
          typename BinaryOperation>
sycl::detail::enable_if_t<(is_group_helper_v<std::decay_t<GroupHelper>>), T>
reduce_over_group(GroupHelper g, V x, T init, BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  return binary_op(init, reduce_over_group(g, x, binary_op));
#else
  (void)g;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

// ---- four reduce_over_group overloads with native binary_op
template <typename GroupHelper, typename T, class BinaryOperation>
sycl::detail::enable_if_t<
    (is_group_helper_v<std::decay_t<GroupHelper>> &&
     sycl::detail::is_complex<T>::value &&
     sycl::detail::is_native_op<T, sycl::plus<T>>::value &&
     sycl::detail::is_plus<T, BinaryOperation>::value),
    T>
reduce_over_group(GroupHelper g, T x, BinaryOperation binary_op) {
  return sycl::reduce_over_group(g.get_group(), x, binary_op);
}

template <typename GroupHelper, typename T, class BinaryOperation>
detail::enable_if_t<(is_group_helper_v<std::decay_t<GroupHelper>> &&
                     sycl::detail::is_vector_arithmetic<T>::value &&
                     sycl::detail::is_native_op<T, BinaryOperation>::value),
                    T>
reduce_over_group(GroupHelper g, T x, BinaryOperation binary_op) {
  return sycl::reduce_over_group(g.get_group(), x, binary_op);
}

template <typename GroupHelper, typename V, typename T, class BinaryOperation>
sycl::detail::enable_if_t<
    (is_group_helper_v<std::decay_t<GroupHelper>> &&
     (sycl::detail::is_scalar_arithmetic<V>::value ||
      sycl::detail::is_complex<V>::value) &&
     (sycl::detail::is_scalar_arithmetic<T>::value ||
      sycl::detail::is_complex<T>::value) &&
     sycl::detail::is_native_op<V, BinaryOperation>::value &&
     sycl::detail::is_native_op<T, BinaryOperation>::value &&
     sycl::detail::is_plus_if_complex<T, BinaryOperation>::value &&
     sycl::detail::is_plus_if_complex<V, BinaryOperation>::value),
    T>
reduce_over_group(GroupHelper g, V x, T init, BinaryOperation binary_op) {
  return sycl::reduce_over_group(g.get_group(), x, binary_op);
}

template <typename GroupHelper, typename V, typename T, class BinaryOperation>
sycl::detail::enable_if_t<
    (is_group_v<std::decay_t<GroupHelper>> &&
     sycl::detail::is_vector_arithmetic<V>::value &&
     sycl::detail::is_vector_arithmetic<T>::value &&
     sycl::detail::is_native_op<V, BinaryOperation>::value &&
     sycl::detail::is_native_op<T, BinaryOperation>::value),
    T>
reduce_over_group(GroupHelper g, V x, T init, BinaryOperation binary_op) {
  return sycl::reduce_over_group(g.get_group(), x, init, binary_op);
}

// ---- joint_reduce
template <typename GroupHelper, typename Ptr, typename BinaryOperation>
sycl::detail::enable_if_t<(is_group_helper_v<std::decay_t<GroupHelper>> &&
                           sycl::detail::is_pointer<Ptr>::value),
                          typename sycl::detail::remove_pointer<Ptr>::type>
joint_reduce(GroupHelper g, Ptr first, Ptr last, BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  using T = typename sycl::detail::remove_pointer<Ptr>::type;
  T init = 0;
  return joint_reduce(g, first, last, init, binary_op);
#else
  (void)g;
  (void)first;
  (void)last;
  (void)binary_op;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename GroupHelper, typename Ptr, typename T,
          typename BinaryOperation>
sycl::detail::enable_if_t<(is_group_helper_v<std::decay_t<GroupHelper>> &&
                           sycl::detail::is_pointer<Ptr>::value),
                          T>
joint_reduce(GroupHelper g, Ptr first, Ptr last, T init,
             BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  T partial;
  sycl::detail::for_each(
      g.get_group(), first, last,
      [&](const typename sycl::detail::remove_pointer<Ptr>::type &x) {
        partial = binary_op(partial, x);
      });
  group_barrier(g.get_group());
  return reduce_over_group(g, partial, init, binary_op);
#else
  (void)g;
  (void)last;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}
} // namespace ext::oneapi::experimental
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
