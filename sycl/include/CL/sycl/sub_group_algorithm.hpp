//==----------- group_algorithm.hpp --- SYCL group algorithm
//---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/__spirv/spirv_ops.hpp>
#include <CL/__spirv/spirv_types.hpp>
#include <CL/__spirv/spirv_vars.hpp>
#include <CL/sycl/ONEAPI/atomic.hpp>
#include <CL/sycl/ONEAPI/functional.hpp>
#include <CL/sycl/detail/spirv.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/group.hpp>
#include <CL/sycl/sub_group.hpp>
#include <CL/sycl/nd_item.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
// ---- get_local_linear_range
template <typename Group> size_t get_local_linear_range(Group g);
template <> inline size_t get_local_linear_range<group<1>>(group<1> g) {
  return g.get_local_range(0);
}
template <> inline size_t get_local_linear_range<group<2>>(group<2> g) {
  return g.get_local_range(0) * g.get_local_range(1);
}
template <> inline size_t get_local_linear_range<group<3>>(group<3> g) {
  return g.get_local_range(0) * g.get_local_range(1) * g.get_local_range(2);
}
template <>
inline size_t get_local_linear_range<ONEAPI::sub_group>(ONEAPI::sub_group g) {
  return g.get_local_range()[0];
}

// ---- get_local_linear_id
template <typename Group>
typename Group::linear_id_type get_local_linear_id(Group g);

#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_GROUP_GET_LOCAL_LINEAR_ID(D)                                    \
  template <>                                                                  \
  group<D>::linear_id_type get_local_linear_id<group<D>>(group<D>) {           \
    nd_item<D> it = cl::sycl::detail::Builder::getNDItem<D>();                 \
    return it.get_local_linear_id();                                           \
  }
__SYCL_GROUP_GET_LOCAL_LINEAR_ID(1);
__SYCL_GROUP_GET_LOCAL_LINEAR_ID(2);
__SYCL_GROUP_GET_LOCAL_LINEAR_ID(3);
#undef __SYCL_GROUP_GET_LOCAL_LINEAR_ID
#endif // __SYCL_DEVICE_ONLY__

template <>
inline ONEAPI::sub_group::linear_id_type
get_local_linear_id<ONEAPI::sub_group>(ONEAPI::sub_group g) {
  return g.get_local_id()[0];
}

// ---- identity
template <typename T, class BinaryOperation> struct identity {};

template <typename T, typename V> struct identity<T, ONEAPI::plus<V>> {
  static constexpr T value = 0;
};

template <typename T, typename V> struct identity<T, ONEAPI::minimum<V>> {
  static constexpr T value = std::numeric_limits<T>::has_infinity
                                 ? std::numeric_limits<T>::infinity()
                                 : (std::numeric_limits<T>::max)();
};

template <typename T, typename V> struct identity<T, ONEAPI::maximum<V>> {
  static constexpr T value =
      std::numeric_limits<T>::has_infinity
          ? static_cast<T>(-std::numeric_limits<T>::infinity())
          : std::numeric_limits<T>::lowest();
};

template <typename T, typename V> struct identity<T, ONEAPI::multiplies<V>> {
  static constexpr T value = static_cast<T>(1);
};

template <typename T, typename V> struct identity<T, ONEAPI::bit_or<V>> {
  static constexpr T value = 0;
};

template <typename T, typename V> struct identity<T, ONEAPI::bit_xor<V>> {
  static constexpr T value = 0;
};

template <typename T, typename V> struct identity<T, ONEAPI::bit_and<V>> {
  static constexpr T value = ~static_cast<T>(0);
};

// ---- is_native_op
template <typename T>
using native_op_list =
    type_list<ONEAPI::plus<T>, ONEAPI::bit_or<T>, ONEAPI::bit_xor<T>,
              ONEAPI::bit_and<T>, ONEAPI::maximum<T>, ONEAPI::minimum<T>,
              ONEAPI::multiplies<T>>;

template <typename T, typename BinaryOperation> struct is_native_op {
  static constexpr bool value =
      is_contained<BinaryOperation, native_op_list<T>>::value ||
      is_contained<BinaryOperation, native_op_list<void>>::value;
};

// ---- for_each
template <typename Group, typename Ptr, class Function>
Function for_each(Group g, Ptr first, Ptr last, Function f) {
#ifdef __SYCL_DEVICE_ONLY__
  ptrdiff_t offset = sycl::detail::get_local_linear_id(g);
  ptrdiff_t stride = sycl::detail::get_local_linear_range(g);
  for (Ptr p = first + offset; p < last; p += stride) {
    f(*p);
  }
  return f;
#else
  (void)g;
  (void)first;
  (void)last;
  (void)f;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}
} // namespace detail

// ---- reduce_over_group
template <typename Group, typename T, class BinaryOperation>
detail::enable_if_t<(is_group_v<Group> &&
                     detail::is_scalar_arithmetic<T>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
reduce_over_group(Group, T x, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(x, x)), T>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(x, x)), float>::value),
      "Result type of binary_op must match reduction accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::calc<T, __spv::GroupOperation::Reduce,
                            sycl::detail::spirv::group_scope<Group>::value>(
      typename sycl::detail::GroupOpTag<T>::type(), x, binary_op);
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, class BinaryOperation>
detail::enable_if_t<(is_group_v<Group> &&
                     detail::is_vector_arithmetic<T>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
reduce_over_group(Group g, T x, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(x[0], x[0])),
                   typename T::element_type>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(x[0], x[0])), float>::value),
      "Result type of binary_op must match reduction accumulation type.");
  T result;
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = reduce_over_group(g, x[s], binary_op);
  }
  return result;
}

template <typename Group, typename T, class BinaryOperation>
detail::enable_if_t<(detail::is_sub_group<Group>::value &&
                     std::is_trivially_copyable<T>::value &&
                     (!detail::is_arithmetic<T>::value ||
                      !detail::is_native_op<T, BinaryOperation>::value)),
                    T>
reduce_over_group(Group g, T x, BinaryOperation op) {
  T result = x;
  for (int mask = 1; mask < g.get_max_local_range()[0]; mask *= 2) {
    T tmp = g.shuffle_xor(result, id<1>(mask));
    if ((g.get_local_id()[0] ^ mask) < g.get_local_range()[0]) {
      result = op(result, tmp);
    }
  }
  return g.shuffle(result, 0);
}

template <typename Group, typename V, typename T, class BinaryOperation>
detail::enable_if_t<(is_group_v<Group> &&
                     detail::is_scalar_arithmetic<V>::value &&
                     detail::is_scalar_arithmetic<T>::value &&
                     detail::is_native_op<V, BinaryOperation>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
reduce_over_group(Group g, V x, T init, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(init, x)), T>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(init, x)), float>::value),
      "Result type of binary_op must match reduction accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  return binary_op(init, reduce_over_group(g, x, binary_op));
#else
  (void)g;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename V, typename T, class BinaryOperation>
detail::enable_if_t<(is_group_v<Group> &&
                     detail::is_vector_arithmetic<V>::value &&
                     detail::is_vector_arithmetic<T>::value &&
                     detail::is_native_op<V, BinaryOperation>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
reduce_over_group(Group g, V x, T init, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(init[0], x[0])),
                   typename T::element_type>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(init[0], x[0])), float>::value),
      "Result type of binary_op must match reduction accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  T result = init;
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = binary_op(init[s], reduce_over_group(g, x[s], binary_op));
  }
  return result;
#else
  (void)g;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename V, typename T, class BinaryOperation>
detail::enable_if_t<(detail::is_sub_group<Group>::value &&
                     std::is_trivially_copyable<T>::value &&
                     std::is_trivially_copyable<V>::value &&
                     (!detail::is_arithmetic<T>::value ||
                      !detail::is_arithmetic<V>::value ||
                      !detail::is_native_op<T, BinaryOperation>::value)),
                    T>
reduce_over_group(Group g, V x, T init, BinaryOperation op) {
  T result = x;
  for (int mask = 1; mask < g.get_max_local_range()[0]; mask *= 2) {
    T tmp = g.shuffle_xor(result, id<1>(mask));
    if ((g.get_local_id()[0] ^ mask) < g.get_local_range()[0]) {
      result = op(result, tmp);
    }
  }
  return g.shuffle(op(init, result), 0);
}

// ---- joint_reduce
template <typename Group, typename Ptr, class BinaryOperation>
detail::enable_if_t<
    (is_group_v<Group> && detail::is_pointer<Ptr>::value &&
     detail::is_arithmetic<typename detail::remove_pointer<Ptr>::type>::value),
    typename detail::remove_pointer<Ptr>::type>
joint_reduce(Group g, Ptr first, Ptr last, BinaryOperation binary_op) {
  using T = typename detail::remove_pointer<Ptr>::type;
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(*first, *first)), T>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(*first, *first)), float>::value),
      "Result type of binary_op must match reduction accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  T partial = sycl::detail::identity<T, BinaryOperation>::value;
  sycl::detail::for_each(g, first, last,
                         [&](const T &x) { partial = binary_op(partial, x); });
  return reduce_over_group(g, partial, binary_op);
#else
  (void)g;
  (void)last;
  (void)binary_op;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename Ptr, typename T, class BinaryOperation>
detail::enable_if_t<
    (is_group_v<Group> && detail::is_pointer<Ptr>::value &&
     detail::is_arithmetic<typename detail::remove_pointer<Ptr>::type>::value &&
     detail::is_arithmetic<T>::value &&
     detail::is_native_op<typename detail::remove_pointer<Ptr>::type,
                          BinaryOperation>::value &&
     detail::is_native_op<T, BinaryOperation>::value),
    T>
joint_reduce(Group g, Ptr first, Ptr last, T init, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(init, *first)), T>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(init, *first)), float>::value),
      "Result type of binary_op must match reduction accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  T partial = sycl::detail::identity<T, BinaryOperation>::value;
  sycl::detail::for_each(
      g, first, last, [&](const typename detail::remove_pointer<Ptr>::type &x) {
        partial = binary_op(partial, x);
      });
  return reduce_over_group(g, partial, init, binary_op);
#else
  (void)g;
  (void)last;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

// ---- any_of_group
template <typename Group>
detail::enable_if_t<is_group_v<Group>, bool> any_of_group(Group, bool pred) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::GroupAny<Group>(pred);
#else
  (void)pred;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, class Predicate>
detail::enable_if_t<is_group_v<Group>, bool> any_of_group(Group g, T x,
                                                          Predicate pred) {
  return any_of_group(g, pred(x));
}

// ---- joint_any_of
template <typename Group, typename Ptr, class Predicate>
detail::enable_if_t<(is_group_v<Group> && detail::is_pointer<Ptr>::value), bool>
joint_any_of(Group g, Ptr first, Ptr last, Predicate pred) {
#ifdef __SYCL_DEVICE_ONLY__
  using T = typename detail::remove_pointer<Ptr>::type;
  bool partial = false;
  sycl::detail::for_each(g, first, last, [&](T &x) { partial |= pred(x); });
  return any_of_group(g, partial);
#else
  (void)g;
  (void)first;
  (void)last;
  (void)pred;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

// ---- all_of_group
template <typename Group>
detail::enable_if_t<is_group_v<Group>, bool> all_of_group(Group, bool pred) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::GroupAll<Group>(pred);
#else
  (void)pred;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, class Predicate>
detail::enable_if_t<is_group_v<Group>, bool> all_of_group(Group g, T x,
                                                          Predicate pred) {
  return all_of_group(g, pred(x));
}

// ---- joint_all_of
template <typename Group, typename Ptr, class Predicate>
detail::enable_if_t<(is_group_v<Group> && detail::is_pointer<Ptr>::value), bool>
joint_all_of(Group g, Ptr first, Ptr last, Predicate pred) {
#ifdef __SYCL_DEVICE_ONLY__
  using T = typename detail::remove_pointer<Ptr>::type;
  bool partial = true;
  sycl::detail::for_each(g, first, last, [&](T &x) { partial &= pred(x); });
  return all_of_group(g, partial);
#else
  (void)g;
  (void)first;
  (void)last;
  (void)pred;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

// ---- none_of_group
template <typename Group>
detail::enable_if_t<is_group_v<Group>, bool> none_of_group(Group, bool pred) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::GroupAll<Group>(!pred);
#else
  (void)pred;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, class Predicate>
detail::enable_if_t<is_group_v<Group>, bool> none_of_group(Group g, T x,
                                                           Predicate pred) {
  return none_of_group(g, pred(x));
}

// ---- joint_none_of
template <typename Group, typename Ptr, class Predicate>
detail::enable_if_t<(is_group_v<Group> && detail::is_pointer<Ptr>::value), bool>
joint_none_of(Group g, Ptr first, Ptr last, Predicate pred) {
#ifdef __SYCL_DEVICE_ONLY__
  return !joint_any_of(g, first, last, pred);
#else
  (void)g;
  (void)first;
  (void)last;
  (void)pred;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
