//==------------------------ group_algorithm.hpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <complex>

#include <CL/__spirv/spirv_ops.hpp>
#include <CL/__spirv/spirv_types.hpp>
#include <CL/__spirv/spirv_vars.hpp>
#include <sycl/builtins.hpp>
#include <sycl/detail/spirv.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/ext/oneapi/functional.hpp>
#include <sycl/functional.hpp>
#include <sycl/group.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/known_identity.hpp>
#include <sycl/nd_item.hpp>
#include <sycl/sub_group.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

// ---- linear_id_to_id
template <int Dimensions>
id<Dimensions> linear_id_to_id(range<Dimensions>, size_t linear_id);
template <> inline id<1> linear_id_to_id(range<1>, size_t linear_id) {
  return id<1>(linear_id);
}
template <> inline id<2> linear_id_to_id(range<2> r, size_t linear_id) {
  id<2> result;
  result[0] = linear_id / r[1];
  result[1] = linear_id % r[1];
  return result;
}
template <> inline id<3> linear_id_to_id(range<3> r, size_t linear_id) {
  id<3> result;
  result[0] = linear_id / (r[1] * r[2]);
  result[1] = (linear_id % (r[1] * r[2])) / r[2];
  result[2] = linear_id % r[2];
  return result;
}

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
inline size_t
get_local_linear_range<ext::oneapi::sub_group>(ext::oneapi::sub_group g) {
  return g.get_local_range()[0];
}

// ---- get_local_linear_id
template <typename Group>
inline typename Group::linear_id_type get_local_linear_id(Group g);

#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_GROUP_GET_LOCAL_LINEAR_ID(D)                                    \
  template <>                                                                  \
  inline group<D>::linear_id_type get_local_linear_id<group<D>>(group<D>) {    \
    nd_item<D> it = sycl::detail::Builder::getNDItem<D>();                     \
    return it.get_local_linear_id();                                           \
  }
__SYCL_GROUP_GET_LOCAL_LINEAR_ID(1);
__SYCL_GROUP_GET_LOCAL_LINEAR_ID(2);
__SYCL_GROUP_GET_LOCAL_LINEAR_ID(3);
#undef __SYCL_GROUP_GET_LOCAL_LINEAR_ID
#endif // __SYCL_DEVICE_ONLY__

template <>
inline ext::oneapi::sub_group::linear_id_type
get_local_linear_id<ext::oneapi::sub_group>(ext::oneapi::sub_group g) {
  return g.get_local_id()[0];
}

// ---- is_native_op
template <typename T>
using native_op_list =
    type_list<sycl::plus<T>, sycl::bit_or<T>, sycl::bit_xor<T>,
              sycl::bit_and<T>, sycl::maximum<T>, sycl::minimum<T>,
              sycl::multiplies<T>>;

template <typename T, typename BinaryOperation> struct is_native_op {
  static constexpr bool value =
      is_contained<BinaryOperation,
                   native_op_list<std::remove_const_t<T>>>::value ||
      is_contained<BinaryOperation,
                   native_op_list<std::add_const_t<T>>>::value ||
      is_contained<BinaryOperation, native_op_list<void>>::value;
};

// ---- is_plus
template <typename T, typename BinaryOperation>
using is_plus = std::integral_constant<
    bool, std::is_same_v<BinaryOperation, sycl::plus<T>> ||
              std::is_same_v<BinaryOperation, sycl::plus<void>>>;

// ---- is_multiplies
template <typename T, typename BinaryOperation>
using is_multiplies = std::integral_constant<
    bool, std::is_same_v<BinaryOperation, sycl::multiplies<T>> ||
              std::is_same_v<BinaryOperation, sycl::multiplies<void>>>;

// ---- is_complex
// NOTE: std::complex<long double> not yet supported by group algorithms.
template <typename T>
struct is_complex
    : std::integral_constant<bool,
                             std::is_same_v<T, std::complex<half>> ||
                                 std::is_same_v<T, std::complex<float>> ||
                                 std::is_same_v<T, std::complex<double>>> {};

// ---- is_arithmetic_or_complex
template <typename T>
using is_arithmetic_or_complex = std::integral_constant<
    bool, sycl::detail::is_complex<typename std::remove_cv_t<T>>::value ||
              sycl::detail::is_arithmetic<T>::value>;

template <typename T>
struct is_vector_arithmetic_or_complex
    : std::bool_constant<is_vec<T>::value &&
                         (is_arithmetic<T>::value ||
                          is_complex<vector_element_t<T>>::value)> {};

// ---- is_plus_or_multiplies_if_complex
template <typename T, typename BinaryOperation>
using is_plus_or_multiplies_if_complex = std::integral_constant<
    bool, (is_complex<T>::value ? (is_plus<T, BinaryOperation>::value ||
                                   is_multiplies<T, BinaryOperation>::value)
                                : std::true_type::value)>;

// ---- identity_for_ga_op
//   the group algorithms support std::complex, limited to sycl::plus operation
//   get the correct identity for group algorithm operation.
// TODO: identiy_for_ga_op should be replaced with known_identity once the other
// callers of known_identity support complex numbers.
template <typename T, class BinaryOperation>
constexpr std::enable_if_t<
    (is_complex<T>::value && is_plus<T, BinaryOperation>::value), T>
identity_for_ga_op() {
  return {0, 0};
}

template <typename T, class BinaryOperation>
constexpr std::enable_if_t<
    (is_complex<T>::value && is_multiplies<T, BinaryOperation>::value), T>
identity_for_ga_op() {
  return {1, 0};
}

template <typename T, class BinaryOperation>
constexpr std::enable_if_t<!is_complex<T>::value, T> identity_for_ga_op() {
  return sycl::known_identity_v<BinaryOperation, T>;
}

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
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}
} // namespace detail

// ---- reduce_over_group
//        three argument variant is specialized thrice:
//        scalar arithmetic, complex (plus only), and vector arithmetic

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                  (detail::is_scalar_arithmetic<T>::value ||
                   (detail::is_complex<T>::value &&
                    detail::is_multiplies<T, BinaryOperation>::value)) &&
                  detail::is_native_op<T, BinaryOperation>::value),
                 T>
reduce_over_group(Group g, T x, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same_v<decltype(binary_op(x, x)), T> ||
          (std::is_same_v<T, half> &&
           std::is_same_v<decltype(binary_op(x, x)), float>),
      "Result type of binary_op must match reduction accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::calc<__spv::GroupOperation::Reduce>(
      g, typename sycl::detail::GroupOpTag<T>::type(), x, binary_op);
#else
  (void)g;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

// complex specialization. T is std::complex<float> or similar.
//   binary op is  sycl::plus<std::complex<float>>
template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                  detail::is_complex<T>::value &&
                  detail::is_native_op<T, sycl::plus<T>>::value &&
                  detail::is_plus<T, BinaryOperation>::value),
                 T>
reduce_over_group(Group g, T x, BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  T result;
  result.real(reduce_over_group(g, x.real(), sycl::plus<>()));
  result.imag(reduce_over_group(g, x.imag(), sycl::plus<>()));
  return result;
#else
  (void)g;
  (void)x;
  (void)binary_op;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, int N, class BinaryOperation>
std::enable_if_t<
    (is_group_v<std::decay_t<Group>> &&
     detail::is_vector_arithmetic_or_complex<sycl::vec<T, N>>::value &&
     detail::is_native_op<sycl::vec<T, N>, BinaryOperation>::value),
    sycl::vec<T, N>>
reduce_over_group(Group g, sycl::vec<T, N> x, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same_v<decltype(binary_op(x[0], x[0])),
                     typename sycl::vec<T, N>::element_type> ||
          (std::is_same_v<sycl::vec<T, N>, half> &&
           std::is_same_v<decltype(binary_op(x[0], x[0])), float>),
      "Result type of binary_op must match reduction accumulation type.");
  sycl::vec<T, N> result;

  detail::loop<N>(
      [&](size_t s) { result[s] = reduce_over_group(g, x[s], binary_op); });
  return result;
}

//   four argument variant of reduce_over_group specialized twice
//       (scalar arithmetic || complex), and vector_arithmetic
template <typename Group, typename V, typename T, class BinaryOperation>
std::enable_if_t<
    (is_group_v<std::decay_t<Group>> &&
     (detail::is_scalar_arithmetic<V>::value || detail::is_complex<V>::value) &&
     (detail::is_scalar_arithmetic<T>::value || detail::is_complex<T>::value) &&
     detail::is_native_op<T, BinaryOperation>::value &&
     detail::is_plus_or_multiplies_if_complex<T, BinaryOperation>::value &&
     std::is_convertible_v<V, T>),
    T>
reduce_over_group(Group g, V x, T init, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same_v<decltype(binary_op(init, x)), T> ||
          (std::is_same_v<T, half> &&
           std::is_same_v<decltype(binary_op(init, x)), float>),
      "Result type of binary_op must match reduction accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  return binary_op(init, reduce_over_group(g, T(x), binary_op));
#else
  (void)g;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename Group, typename V, typename T, class BinaryOperation>
std::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                  detail::is_vector_arithmetic_or_complex<V>::value &&
                  detail::is_vector_arithmetic_or_complex<T>::value &&
                  detail::is_native_op<V, BinaryOperation>::value &&
                  detail::is_native_op<T, BinaryOperation>::value),
                 T>
reduce_over_group(Group g, V x, T init, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same_v<decltype(binary_op(init[0], x[0])),
                     typename T::element_type> ||
          (std::is_same_v<T, half> &&
           std::is_same_v<decltype(binary_op(init[0], x[0])), float>),
      "Result type of binary_op must match reduction accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  T result = init;
  for (int s = 0; s < x.size(); ++s) {
    result[s] = binary_op(init[s], reduce_over_group(g, x[s], binary_op));
  }
  return result;
#else
  (void)g;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

// ---- joint_reduce
template <typename Group, typename Ptr, typename T, class BinaryOperation>
std::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer<Ptr>::value &&
     detail::is_arithmetic_or_complex<
         typename detail::remove_pointer<Ptr>::type>::value &&
     detail::is_arithmetic_or_complex<T>::value &&
     detail::is_plus_or_multiplies_if_complex<T, BinaryOperation>::value &&
     detail::is_native_op<T, BinaryOperation>::value),
    T>
joint_reduce(Group g, Ptr first, Ptr last, T init, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same_v<decltype(binary_op(init, *first)), T> ||
          (std::is_same_v<T, half> &&
           std::is_same_v<decltype(binary_op(init, *first)), float>),
      "Result type of binary_op must match reduction accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  T partial = detail::identity_for_ga_op<T, BinaryOperation>();
  sycl::detail::for_each(
      g, first, last, [&](const typename detail::remove_pointer<Ptr>::type &x) {
        partial = binary_op(partial, x);
      });
  return reduce_over_group(g, partial, init, binary_op);
#else
  (void)g;
  (void)last;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename Group, typename Ptr, class BinaryOperation>
std::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer<Ptr>::value &&
     detail::is_arithmetic_or_complex<
         typename detail::remove_pointer<Ptr>::type>::value &&
     detail::is_plus_or_multiplies_if_complex<
         typename detail::remove_pointer<Ptr>::type, BinaryOperation>::value),
    typename detail::remove_pointer<Ptr>::type>
joint_reduce(Group g, Ptr first, Ptr last, BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  using T = typename detail::remove_pointer<Ptr>::type;
  T init = detail::identity_for_ga_op<T, BinaryOperation>();
  return joint_reduce(g, first, last, init, binary_op);
#else
  (void)g;
  (void)first;
  (void)last;
  (void)binary_op;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

// ---- any_of_group
template <typename Group>
std::enable_if_t<is_group_v<std::decay_t<Group>>, bool>
any_of_group(Group g, bool pred) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::GroupAny(g, pred);
#else
  (void)g;
  (void)pred;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, class Predicate>
std::enable_if_t<is_group_v<Group>, bool> any_of_group(Group g, T x,
                                                       Predicate pred) {
  return any_of_group(g, pred(x));
}

// ---- joint_any_of
template <typename Group, typename Ptr, class Predicate>
std::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer<Ptr>::value), bool>
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
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

// ---- all_of_group
template <typename Group>
std::enable_if_t<is_group_v<std::decay_t<Group>>, bool>
all_of_group(Group g, bool pred) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::GroupAll(g, pred);
#else
  (void)g;
  (void)pred;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, class Predicate>
std::enable_if_t<is_group_v<std::decay_t<Group>>, bool>
all_of_group(Group g, T x, Predicate pred) {
  return all_of_group(g, pred(x));
}

// ---- joint_all_of
template <typename Group, typename Ptr, class Predicate>
std::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer<Ptr>::value), bool>
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
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

// ---- none_of_group
template <typename Group>
std::enable_if_t<is_group_v<std::decay_t<Group>>, bool>
none_of_group(Group g, bool pred) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::GroupAll(g, !pred);
#else
  (void)g;
  (void)pred;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, class Predicate>
std::enable_if_t<is_group_v<std::decay_t<Group>>, bool>
none_of_group(Group g, T x, Predicate pred) {
  return none_of_group(g, pred(x));
}

// ---- joint_none_of
template <typename Group, typename Ptr, class Predicate>
std::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer<Ptr>::value), bool>
joint_none_of(Group g, Ptr first, Ptr last, Predicate pred) {
#ifdef __SYCL_DEVICE_ONLY__
  return !joint_any_of(g, first, last, pred);
#else
  (void)g;
  (void)first;
  (void)last;
  (void)pred;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

// ---- shift_group_left
// TODO: remove check for detail::is_vec<T> once sycl::vec is trivially
// copyable.
template <typename Group, typename T>
std::enable_if_t<(std::is_same_v<std::decay_t<Group>, sub_group> &&
                  (std::is_trivially_copyable_v<T> ||
                   detail::is_vec<T>::value)),
                 T>
shift_group_left(Group, T x, typename Group::linear_id_type delta = 1) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::SubgroupShuffleDown(x, delta);
#else
  (void)x;
  (void)delta;
  throw runtime_error("Sub-groups are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

// ---- shift_group_right
// TODO: remove check for detail::is_vec<T> once sycl::vec is trivially
// copyable.
template <typename Group, typename T>
std::enable_if_t<(std::is_same_v<std::decay_t<Group>, sub_group> &&
                  (std::is_trivially_copyable_v<T> ||
                   detail::is_vec<T>::value)),
                 T>
shift_group_right(Group, T x, typename Group::linear_id_type delta = 1) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::SubgroupShuffleUp(x, delta);
#else
  (void)x;
  (void)delta;
  throw runtime_error("Sub-groups are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

// ---- permute_group_by_xor
// TODO: remove check for detail::is_vec<T> once sycl::vec is trivially
// copyable.
template <typename Group, typename T>
std::enable_if_t<(std::is_same_v<std::decay_t<Group>, sub_group> &&
                  (std::is_trivially_copyable_v<T> ||
                   detail::is_vec<T>::value)),
                 T>
permute_group_by_xor(Group, T x, typename Group::linear_id_type mask) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::SubgroupShuffleXor(x, mask);
#else
  (void)x;
  (void)mask;
  throw runtime_error("Sub-groups are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

// ---- select_from_group
// TODO: remove check for detail::is_vec<T> once sycl::vec is trivially
// copyable.
template <typename Group, typename T>
std::enable_if_t<(std::is_same_v<std::decay_t<Group>, sub_group> &&
                  (std::is_trivially_copyable_v<T> ||
                   detail::is_vec<T>::value)),
                 T>
select_from_group(Group, T x, typename Group::id_type local_id) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::SubgroupShuffle(x, local_id);
#else
  (void)x;
  (void)local_id;
  throw runtime_error("Sub-groups are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

// ---- group_broadcast
// TODO: remove check for detail::is_vec<T> once sycl::vec is trivially
// copyable.
template <typename Group, typename T>
std::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                  (std::is_trivially_copyable_v<T> ||
                   detail::is_vec<T>::value)),
                 T>
group_broadcast(Group g, T x, typename Group::id_type local_id) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::GroupBroadcast(g, x, local_id);
#else
  (void)g;
  (void)x;
  (void)local_id;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename Group, typename T>
std::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                  (std::is_trivially_copyable_v<T> ||
                   detail::is_vec<T>::value)),
                 T>
group_broadcast(Group g, T x, typename Group::linear_id_type linear_local_id) {
#ifdef __SYCL_DEVICE_ONLY__
  return group_broadcast(
      g, x,
      sycl::detail::linear_id_to_id(g.get_local_range(), linear_local_id));
#else
  (void)g;
  (void)x;
  (void)linear_local_id;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename Group, typename T>
std::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                  (std::is_trivially_copyable_v<T> ||
                   detail::is_vec<T>::value)),
                 T>
group_broadcast(Group g, T x) {
#ifdef __SYCL_DEVICE_ONLY__
  return group_broadcast(g, x, 0);
#else
  (void)g;
  (void)x;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

// ---- exclusive_scan_over_group
//   this function has two overloads, one with three arguments and one with four
//   the three argument version is specialized thrice: scalar, complex, and
//   vector
template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                  (detail::is_scalar_arithmetic<T>::value ||
                   (detail::is_complex<T>::value &&
                    detail::is_multiplies<T, BinaryOperation>::value)) &&
                  detail::is_native_op<T, BinaryOperation>::value),
                 T>
exclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(std::is_same_v<decltype(binary_op(x, x)), T> ||
                    (std::is_same_v<T, half> &&
                     std::is_same_v<decltype(binary_op(x, x)), float>),
                "Result type of binary_op must match scan accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::calc<__spv::GroupOperation::ExclusiveScan>(
      g, typename sycl::detail::GroupOpTag<T>::type(), x, binary_op);
#else
  (void)g;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

// complex specialization. T is std::complex<float> or similar.
//   binary op is  sycl::plus<std::complex<float>>
template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                  detail::is_complex<T>::value &&
                  detail::is_native_op<T, sycl::plus<T>>::value &&
                  detail::is_plus<T, BinaryOperation>::value),
                 T>
exclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  T result;
  result.real(exclusive_scan_over_group(g, x.real(), sycl::plus<>()));
  result.imag(exclusive_scan_over_group(g, x.imag(), sycl::plus<>()));
  return result;
#else
  (void)g;
  (void)x;
  (void)binary_op;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                  detail::is_vector_arithmetic_or_complex<T>::value &&
                  detail::is_native_op<T, BinaryOperation>::value),
                 T>
exclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(std::is_same_v<decltype(binary_op(x[0], x[0])),
                               typename T::element_type> ||
                    (std::is_same_v<T, half> &&
                     std::is_same_v<decltype(binary_op(x[0], x[0])), float>),
                "Result type of binary_op must match scan accumulation type.");
  T result;
  for (int s = 0; s < x.size(); ++s) {
    result[s] = exclusive_scan_over_group(g, x[s], binary_op);
  }
  return result;
}

// four argument version of exclusive_scan_over_group is specialized twice
// once for vector_arithmetic, once for (scalar_arithmetic || complex)
template <typename Group, typename V, typename T, class BinaryOperation>
std::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                  detail::is_vector_arithmetic_or_complex<V>::value &&
                  detail::is_vector_arithmetic_or_complex<T>::value &&
                  detail::is_native_op<V, BinaryOperation>::value &&
                  detail::is_native_op<T, BinaryOperation>::value),
                 T>
exclusive_scan_over_group(Group g, V x, T init, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(std::is_same_v<decltype(binary_op(init[0], x[0])),
                               typename T::element_type> ||
                    (std::is_same_v<T, half> &&
                     std::is_same_v<decltype(binary_op(init[0], x[0])), float>),
                "Result type of binary_op must match scan accumulation type.");
  T result;
  for (int s = 0; s < x.size(); ++s) {
    result[s] = exclusive_scan_over_group(g, x[s], init[s], binary_op);
  }
  return result;
}

template <typename Group, typename V, typename T, class BinaryOperation>
std::enable_if_t<
    (is_group_v<std::decay_t<Group>> &&
     (detail::is_scalar_arithmetic<V>::value || detail::is_complex<V>::value) &&
     (detail::is_scalar_arithmetic<T>::value || detail::is_complex<T>::value) &&
     detail::is_native_op<T, BinaryOperation>::value &&
     detail::is_plus_or_multiplies_if_complex<T, BinaryOperation>::value &&
     std::is_convertible_v<V, T>),
    T>
exclusive_scan_over_group(Group g, V x, T init, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(std::is_same_v<decltype(binary_op(init, x)), T> ||
                    (std::is_same_v<T, half> &&
                     std::is_same_v<decltype(binary_op(init, x)), float>),
                "Result type of binary_op must match scan accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  typename Group::linear_id_type local_linear_id =
      sycl::detail::get_local_linear_id(g);
  T y = x;
  if (local_linear_id == 0) {
    y = binary_op(init, y);
  }
  T scan = exclusive_scan_over_group(g, y, binary_op);
  if (local_linear_id == 0) {
    scan = init;
  }
  return scan;
#else
  (void)g;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

// ---- joint_exclusive_scan
template <typename Group, typename InPtr, typename OutPtr, typename T,
          class BinaryOperation>
std::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer<InPtr>::value &&
     detail::is_pointer<OutPtr>::value &&
     detail::is_arithmetic_or_complex<
         typename detail::remove_pointer<InPtr>::type>::value &&
     detail::is_arithmetic_or_complex<
         typename detail::remove_pointer<OutPtr>::type>::value &&
     detail::is_arithmetic_or_complex<T>::value &&
     detail::is_native_op<T, BinaryOperation>::value &&
     detail::is_plus_or_multiplies_if_complex<T, BinaryOperation>::value),
    OutPtr>
joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result, T init,
                     BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(std::is_same_v<decltype(binary_op(init, *first)), T> ||
                    (std::is_same_v<T, half> &&
                     std::is_same_v<decltype(binary_op(init, *first)), float>),
                "Result type of binary_op must match scan accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  ptrdiff_t offset = sycl::detail::get_local_linear_id(g);
  ptrdiff_t stride = sycl::detail::get_local_linear_range(g);
  ptrdiff_t N = last - first;
  auto roundup = [=](const ptrdiff_t &v,
                     const ptrdiff_t &divisor) -> ptrdiff_t {
    return ((v + divisor - 1) / divisor) * divisor;
  };
  typename std::remove_const<typename detail::remove_pointer<InPtr>::type>::type
      x;
  T carry = init;
  for (ptrdiff_t chunk = 0; chunk < roundup(N, stride); chunk += stride) {
    ptrdiff_t i = chunk + offset;
    if (i < N) {
      x = first[i];
    }
    T out = exclusive_scan_over_group(g, x, carry, binary_op);
    if (i < N) {
      result[i] = out;
    }
    carry = group_broadcast(g, binary_op(out, x), stride - 1);
  }
  return result + N;
#else
  (void)g;
  (void)last;
  (void)result;
  (void)init;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation>
std::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer<InPtr>::value &&
     detail::is_pointer<OutPtr>::value &&
     detail::is_arithmetic_or_complex<
         typename detail::remove_pointer<InPtr>::type>::value &&
     detail::is_arithmetic_or_complex<
         typename detail::remove_pointer<OutPtr>::type>::value &&
     detail::is_native_op<typename detail::remove_pointer<OutPtr>::type,
                          BinaryOperation>::value &&
     detail::is_plus_or_multiplies_if_complex<
         typename detail::remove_pointer<OutPtr>::type,
         BinaryOperation>::value),
    OutPtr>
joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                     BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same_v<decltype(binary_op(*first, *first)),
                     typename detail::remove_pointer<OutPtr>::type> ||
          (std::is_same_v<typename detail::remove_pointer<OutPtr>::type,
                          half> &&
           std::is_same_v<decltype(binary_op(*first, *first)), float>),
      "Result type of binary_op must match scan accumulation type.");
  using T = typename detail::remove_pointer<OutPtr>::type;
  T init = detail::identity_for_ga_op<T, BinaryOperation>();
  return joint_exclusive_scan(g, first, last, result, init, binary_op);
}

// ---- inclusive_scan_over_group
//   this function has two overloads, one with three arguments and one with four
//   the three argument version is specialized thrice: vector, scalar, and
//   complex
template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                  detail::is_vector_arithmetic_or_complex<T>::value &&
                  detail::is_native_op<T, BinaryOperation>::value),
                 T>
inclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(std::is_same_v<decltype(binary_op(x[0], x[0])),
                               typename T::element_type> ||
                    (std::is_same_v<T, half> &&
                     std::is_same_v<decltype(binary_op(x[0], x[0])), float>),
                "Result type of binary_op must match scan accumulation type.");
  T result;
  for (int s = 0; s < x.size(); ++s) {
    result[s] = inclusive_scan_over_group(g, x[s], binary_op);
  }
  return result;
}

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                  (detail::is_scalar_arithmetic<T>::value ||
                   (detail::is_complex<T>::value &&
                    detail::is_multiplies<T, BinaryOperation>::value)) &&
                  detail::is_native_op<T, BinaryOperation>::value),
                 T>
inclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(std::is_same_v<decltype(binary_op(x, x)), T> ||
                    (std::is_same_v<T, half> &&
                     std::is_same_v<decltype(binary_op(x, x)), float>),
                "Result type of binary_op must match scan accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::calc<__spv::GroupOperation::InclusiveScan>(
      g, typename sycl::detail::GroupOpTag<T>::type(), x, binary_op);
#else
  (void)g;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

// complex specializaiton
template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                  detail::is_complex<T>::value &&
                  detail::is_native_op<T, sycl::plus<T>>::value &&
                  detail::is_plus<T, BinaryOperation>::value),
                 T>
inclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
#ifdef __SYCL_DEVICE_ONLY__
  T result;
  result.real(inclusive_scan_over_group(g, x.real(), sycl::plus<>()));
  result.imag(inclusive_scan_over_group(g, x.imag(), sycl::plus<>()));
  return result;
#else
  (void)g;
  (void)x;
  (void)binary_op;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

// four argument version of inclusive_scan_over_group is specialized twice
// once for (scalar_arithmetic || complex) and once for vector_arithmetic
template <typename Group, typename V, class BinaryOperation, typename T>
std::enable_if_t<
    (is_group_v<std::decay_t<Group>> &&
     (detail::is_scalar_arithmetic<V>::value || detail::is_complex<V>::value) &&
     (detail::is_scalar_arithmetic<T>::value || detail::is_complex<T>::value) &&
     detail::is_native_op<T, BinaryOperation>::value &&
     detail::is_plus_or_multiplies_if_complex<T, BinaryOperation>::value &&
     std::is_convertible_v<V, T>),
    T>
inclusive_scan_over_group(Group g, V x, BinaryOperation binary_op, T init) {
  // FIXME: Do not special-case for half precision
  static_assert(std::is_same_v<decltype(binary_op(init, x)), T> ||
                    (std::is_same_v<T, half> &&
                     std::is_same_v<decltype(binary_op(init, x)), float>),
                "Result type of binary_op must match scan accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  T y = x;
  if (sycl::detail::get_local_linear_id(g) == 0) {
    y = binary_op(init, y);
  }
  return inclusive_scan_over_group(g, y, binary_op);
#else
  (void)g;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename Group, typename V, class BinaryOperation, typename T>
std::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                  detail::is_vector_arithmetic_or_complex<V>::value &&
                  detail::is_vector_arithmetic_or_complex<T>::value &&
                  detail::is_native_op<V, BinaryOperation>::value &&
                  detail::is_native_op<T, BinaryOperation>::value),
                 T>
inclusive_scan_over_group(Group g, V x, BinaryOperation binary_op, T init) {
  // FIXME: Do not special-case for half precision
  static_assert(std::is_same_v<decltype(binary_op(init[0], x[0])), T> ||
                    (std::is_same_v<T, half> &&
                     std::is_same_v<decltype(binary_op(init[0], x[0])), float>),
                "Result type of binary_op must match scan accumulation type.");
  T result;
  for (int s = 0; s < x.size(); ++s) {
    result[s] = inclusive_scan_over_group(g, x[s], binary_op, init[s]);
  }
  return result;
}

// ---- joint_inclusive_scan
template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation, typename T>
std::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer<InPtr>::value &&
     detail::is_pointer<OutPtr>::value &&
     detail::is_arithmetic_or_complex<
         typename detail::remove_pointer<InPtr>::type>::value &&
     detail::is_arithmetic_or_complex<
         typename detail::remove_pointer<OutPtr>::type>::value &&
     detail::is_arithmetic_or_complex<T>::value &&
     detail::is_native_op<T, BinaryOperation>::value &&
     detail::is_plus_or_multiplies_if_complex<T, BinaryOperation>::value),
    OutPtr>
joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                     BinaryOperation binary_op, T init) {
  // FIXME: Do not special-case for half precision
  static_assert(std::is_same_v<decltype(binary_op(init, *first)), T> ||
                    (std::is_same_v<T, half> &&
                     std::is_same_v<decltype(binary_op(init, *first)), float>),
                "Result type of binary_op must match scan accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  ptrdiff_t offset = sycl::detail::get_local_linear_id(g);
  ptrdiff_t stride = sycl::detail::get_local_linear_range(g);
  ptrdiff_t N = last - first;
  auto roundup = [=](const ptrdiff_t &v,
                     const ptrdiff_t &divisor) -> ptrdiff_t {
    return ((v + divisor - 1) / divisor) * divisor;
  };
  typename std::remove_const<typename detail::remove_pointer<InPtr>::type>::type
      x;
  T carry = init;
  for (ptrdiff_t chunk = 0; chunk < roundup(N, stride); chunk += stride) {
    ptrdiff_t i = chunk + offset;
    if (i < N) {
      x = first[i];
    }
    T out = inclusive_scan_over_group(g, x, binary_op, carry);
    if (i < N) {
      result[i] = out;
    }
    carry = group_broadcast(g, out, stride - 1);
  }
  return result + N;
#else
  (void)g;
  (void)last;
  (void)result;
  throw runtime_error("Group algorithms are not supported on host.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation>
std::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer<InPtr>::value &&
     detail::is_pointer<OutPtr>::value &&
     detail::is_arithmetic_or_complex<
         typename detail::remove_pointer<InPtr>::type>::value &&
     detail::is_native_op<typename detail::remove_pointer<OutPtr>::type,
                          BinaryOperation>::value &&
     detail::is_plus_or_multiplies_if_complex<
         typename detail::remove_pointer<OutPtr>::type,
         BinaryOperation>::value),
    OutPtr>
joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                     BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same_v<decltype(binary_op(*first, *first)),
                     typename detail::remove_pointer<OutPtr>::type> ||
          (std::is_same_v<typename detail::remove_pointer<OutPtr>::type,
                          half> &&
           std::is_same_v<decltype(binary_op(*first, *first)), float>),
      "Result type of binary_op must match scan accumulation type.");

  using T = typename detail::remove_pointer<OutPtr>::type;
  T init = detail::identity_for_ga_op<T, BinaryOperation>();
  return joint_inclusive_scan(g, first, last, result, binary_op, init);
}

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
