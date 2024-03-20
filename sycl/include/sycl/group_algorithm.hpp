//==------------------------ group_algorithm.hpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/array.hpp>       // for array
#include <sycl/detail/helpers.hpp>     // for loop
#include <sycl/detail/item_base.hpp>   // for id, range
#include <sycl/detail/type_list.hpp>   // for is_contained, type_list
#include <sycl/detail/type_traits.hpp> // for remove_pointer, is_pointer
#include <sycl/exception.hpp>          // for make_error_code, errc, exception
#include <sycl/functional.hpp>         // for plus, multiplies, maximum
#include <sycl/group.hpp>              // for group
#include <sycl/half_type.hpp>          // for half
#include <sycl/id.hpp>                 // for id
#include <sycl/known_identity.hpp>     // for known_identity_v
#include <sycl/nd_item.hpp>            // for nd_item
#include <sycl/range.hpp>              // for range
#include <sycl/sub_group.hpp>          // for sub_group
#include <sycl/types.hpp>              // for vec

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/ext/oneapi/functional.hpp>
#if defined(__NVPTX__)
#include <sycl/ext/oneapi/experimental/cuda/non_uniform_algorithms.hpp>
#endif
#endif

#include <stddef.h>    // for size_t
#include <type_traits> // for enable_if_t, decay_t, integra...

namespace sycl {
inline namespace _V1 {
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
template <typename Group> inline auto get_local_linear_range(Group g) {
  auto local_range = g.get_local_range();
  auto result = local_range[0];
  for (size_t i = 1; i < Group::dimensions; ++i)
    result *= local_range[i];
  return result;
}

// ---- get_local_linear_id
template <typename Group> inline auto get_local_linear_id(Group g) {
#ifdef __SYCL_DEVICE_ONLY__
  if constexpr (std::is_same_v<Group, group<1>> ||
                std::is_same_v<Group, group<2>> ||
                std::is_same_v<Group, group<3>>) {
    auto it = sycl::detail::Builder::getNDItem<Group::dimensions>();
    return it.get_local_linear_id();
  }
#endif // __SYCL_DEVICE_ONLY__
  return g.get_local_linear_id();
}

// ---- is_native_op
template <typename T>
using native_op_list =
    type_list<sycl::plus<T>, sycl::bit_or<T>, sycl::bit_xor<T>,
              sycl::bit_and<T>, sycl::maximum<T>, sycl::minimum<T>,
              sycl::multiplies<T>, sycl::logical_or<T>, sycl::logical_and<T>>;

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
    bool,
    std::is_same_v<BinaryOperation, sycl::plus<std::remove_const_t<T>>> ||
        std::is_same_v<BinaryOperation, sycl::plus<std::add_const_t<T>>> ||
        std::is_same_v<BinaryOperation, sycl::plus<void>>>;

// ---- is_multiplies
template <typename T, typename BinaryOperation>
using is_multiplies = std::integral_constant<
    bool,
    std::is_same_v<BinaryOperation, sycl::multiplies<std::remove_const_t<T>>> ||
        std::is_same_v<BinaryOperation,
                       sycl::multiplies<std::add_const_t<T>>> ||
        std::is_same_v<BinaryOperation, sycl::multiplies<void>>>;

// ---- is_complex
// Use SFINAE so that the "true" branch could be implemented in
// include/sycl/stl_wrappers/complex that would only be available if STL's
// <complex> is included by users.
template <typename T, typename = void>
struct is_complex : public std::false_type {};

// ---- is_arithmetic_or_complex
template <typename T>
using is_arithmetic_or_complex =
    std::integral_constant<bool, sycl::detail::is_complex<T>::value ||
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

// used to transform a vector op to a scalar op;
// e.g. sycl::plus<std::vec<T, N>> to sycl::plus<T>
template <typename T> struct get_scalar_binary_op;

template <template <typename> typename F, typename T, int n>
struct get_scalar_binary_op<F<sycl::vec<T, n>>> {
  using type = F<T>;
};

template <template <typename> typename F> struct get_scalar_binary_op<F<void>> {
  using type = F<void>;
};

// ---- is_max_or_min
template <typename T> struct is_max_or_min : std::false_type {};
template <typename T>
struct is_max_or_min<sycl::maximum<T>> : std::true_type {};
template <typename T>
struct is_max_or_min<sycl::minimum<T>> : std::true_type {};

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
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
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
  static_assert(
      std::is_same_v<decltype(binary_op(x, x)), T>,
      "Result type of binary_op must match reduction accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
  if constexpr (ext::oneapi::experimental::is_user_constructed_group_v<Group>) {
    sycl::vec<unsigned, 4> MemberMask =
        sycl::detail::ExtractMask(sycl::detail::GetMask(g));
#if (__SYCL_CUDA_ARCH__ >= 800)
    return detail::masked_reduction_cuda_sm80(g, x, binary_op, MemberMask[0]);
#else
    return detail::masked_reduction_cuda_shfls(g, x, binary_op, MemberMask[0]);
#endif
  }
#endif
  return sycl::detail::calc<__spv::GroupOperation::Reduce>(
      g, typename sycl::detail::GroupOpTag<T>::type(), x, binary_op);
#else
  (void)g;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
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
reduce_over_group(Group g, T x, BinaryOperation) {
#ifdef __SYCL_DEVICE_ONLY__
  T result;
  result.real(reduce_over_group(g, x.real(), sycl::plus<>()));
  result.imag(reduce_over_group(g, x.imag(), sycl::plus<>()));
  return result;
#else
  (void)g;
  (void)x;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
#endif
}

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                  detail::is_vector_arithmetic_or_complex<T>::value &&
                  detail::is_native_op<T, BinaryOperation>::value),
                 T>
reduce_over_group(Group g, T x, BinaryOperation binary_op) {
  static_assert(
      std::is_same_v<decltype(binary_op(x, x)), T>,
      "Result type of binary_op must match reduction accumulation type.");
  T result;
  typename detail::get_scalar_binary_op<BinaryOperation>::type
      scalar_binary_op{};
  detail::loop<x.size()>([&](size_t s) {
    result[s] = reduce_over_group(g, x[s], scalar_binary_op);
  });
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
  static_assert(
      std::is_same_v<decltype(binary_op(init, x)), T>,
      "Result type of binary_op must match reduction accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  return binary_op(init, reduce_over_group(g, T(x), binary_op));
#else
  (void)g;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
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
  static_assert(
      std::is_same_v<decltype(binary_op(init, x)), T>,
      "Result type of binary_op must match reduction accumulation type.");
  typename detail::get_scalar_binary_op<BinaryOperation>::type
      scalar_binary_op{};
#ifdef __SYCL_DEVICE_ONLY__
  T result = init;
  for (int s = 0; s < x.size(); ++s) {
    result[s] =
        scalar_binary_op(init[s], reduce_over_group(g, x[s], scalar_binary_op));
  }
  return result;
#else
  (void)g;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
#endif
}

// ---- joint_reduce
template <typename Group, typename Ptr, typename T, class BinaryOperation>
std::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer_v<Ptr> &&
     detail::is_arithmetic_or_complex<
         typename detail::remove_pointer<Ptr>::type>::value &&
     detail::is_arithmetic_or_complex<T>::value &&
     detail::is_plus_or_multiplies_if_complex<T, BinaryOperation>::value &&
     detail::is_native_op<T, BinaryOperation>::value),
    T>
joint_reduce(Group g, Ptr first, Ptr last, T init, BinaryOperation binary_op) {
  static_assert(
      std::is_same_v<decltype(binary_op(init, *first)), T>,
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
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
#endif
}

template <typename Group, typename Ptr, class BinaryOperation>
std::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer_v<Ptr> &&
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
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
#endif
}

// ---- any_of_group
template <typename Group>
std::enable_if_t<is_group_v<std::decay_t<Group>>, bool>
any_of_group(Group g, bool pred) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
  if constexpr (ext::oneapi::experimental::is_user_constructed_group_v<Group>) {
    return __nvvm_vote_any_sync(detail::ExtractMask(detail::GetMask(g))[0],
                                pred);
  }
#endif
  return sycl::detail::spirv::GroupAny(g, pred);
#else
  (void)g;
  (void)pred;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
#endif
}

template <typename Group, typename T, class Predicate>
std::enable_if_t<is_group_v<Group>, bool> any_of_group(Group g, T x,
                                                       Predicate pred) {
  return any_of_group(g, pred(x));
}

// ---- joint_any_of
template <typename Group, typename Ptr, class Predicate>
std::enable_if_t<(is_group_v<std::decay_t<Group>> && detail::is_pointer_v<Ptr>),
                 bool>
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
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
#endif
}

// ---- all_of_group
template <typename Group>
std::enable_if_t<is_group_v<std::decay_t<Group>>, bool>
all_of_group(Group g, bool pred) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
  if constexpr (ext::oneapi::experimental::is_user_constructed_group_v<Group>) {
    return __nvvm_vote_all_sync(detail::ExtractMask(detail::GetMask(g))[0],
                                pred);
  }
#endif
  return sycl::detail::spirv::GroupAll(g, pred);
#else
  (void)g;
  (void)pred;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
#endif
}

template <typename Group, typename T, class Predicate>
std::enable_if_t<is_group_v<std::decay_t<Group>>, bool>
all_of_group(Group g, T x, Predicate pred) {
  return all_of_group(g, pred(x));
}

// ---- joint_all_of
template <typename Group, typename Ptr, class Predicate>
std::enable_if_t<(is_group_v<std::decay_t<Group>> && detail::is_pointer_v<Ptr>),
                 bool>
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
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
#endif
}

// ---- none_of_group
template <typename Group>
std::enable_if_t<is_group_v<std::decay_t<Group>>, bool>
none_of_group(Group g, bool pred) {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
  if constexpr (ext::oneapi::experimental::is_user_constructed_group_v<Group>) {
    return __nvvm_vote_all_sync(detail::ExtractMask(detail::GetMask(g))[0],
                                !pred);
  }
#endif
  return sycl::detail::spirv::GroupAll(g, !pred);
#else
  (void)g;
  (void)pred;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
#endif
}

template <typename Group, typename T, class Predicate>
std::enable_if_t<is_group_v<std::decay_t<Group>>, bool>
none_of_group(Group g, T x, Predicate pred) {
  return none_of_group(g, pred(x));
}

// ---- joint_none_of
template <typename Group, typename Ptr, class Predicate>
std::enable_if_t<(is_group_v<std::decay_t<Group>> && detail::is_pointer_v<Ptr>),
                 bool>
joint_none_of(Group g, Ptr first, Ptr last, Predicate pred) {
#ifdef __SYCL_DEVICE_ONLY__
  return !joint_any_of(g, first, last, pred);
#else
  (void)g;
  (void)first;
  (void)last;
  (void)pred;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
#endif
}

// ---- shift_group_left
// TODO: remove check for detail::is_vec<T> once sycl::vec is trivially
// copyable.
template <typename Group, typename T>
std::enable_if_t<((std::is_same_v<std::decay_t<Group>, sub_group> ||
                   sycl::ext::oneapi::experimental::is_user_constructed_group_v<
                       std::decay_t<Group>>) &&
                  (std::is_trivially_copyable_v<T> ||
                   detail::is_vec<T>::value)),
                 T>
shift_group_left(Group g, T x, typename Group::linear_id_type delta = 1) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::ShuffleDown(g, x, delta);
#else
  (void)g;
  (void)x;
  (void)delta;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

// ---- shift_group_right
// TODO: remove check for detail::is_vec<T> once sycl::vec is trivially
// copyable.
template <typename Group, typename T>
std::enable_if_t<((std::is_same_v<std::decay_t<Group>, sub_group> ||
                   sycl::ext::oneapi::experimental::is_user_constructed_group_v<
                       std::decay_t<Group>>) &&
                  (std::is_trivially_copyable_v<T> ||
                   detail::is_vec<T>::value)),
                 T>
shift_group_right(Group g, T x, typename Group::linear_id_type delta = 1) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::ShuffleUp(g, x, delta);
#else
  (void)g;
  (void)x;
  (void)delta;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

// ---- permute_group_by_xor
// TODO: remove check for detail::is_vec<T> once sycl::vec is trivially
// copyable.
template <typename Group, typename T>
std::enable_if_t<((std::is_same_v<std::decay_t<Group>, sub_group> ||
                   sycl::ext::oneapi::experimental::is_user_constructed_group_v<
                       std::decay_t<Group>>) &&
                  (std::is_trivially_copyable_v<T> ||
                   detail::is_vec<T>::value)),
                 T>
permute_group_by_xor(Group g, T x, typename Group::linear_id_type mask) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::ShuffleXor(g, x, mask);
#else
  (void)g;
  (void)x;
  (void)mask;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

// ---- select_from_group
// TODO: remove check for detail::is_vec<T> once sycl::vec is trivially
// copyable.
template <typename Group, typename T>
std::enable_if_t<((std::is_same_v<std::decay_t<Group>, sub_group> ||
                   sycl::ext::oneapi::experimental::is_user_constructed_group_v<
                       std::decay_t<Group>>) &&
                  (std::is_trivially_copyable_v<T> ||
                   detail::is_vec<T>::value)),
                 T>
select_from_group(Group g, T x, typename Group::id_type local_id) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::Shuffle(g, x, local_id);
#else
  (void)g;
  (void)x;
  (void)local_id;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
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
#if defined(__NVPTX__)
  if constexpr (ext::oneapi::experimental::is_user_constructed_group_v<Group>) {
    auto LocalId = detail::IdToMaskPosition(g, local_id);
    return __nvvm_shfl_sync_idx_i32(detail::ExtractMask(detail::GetMask(g))[0],
                                    x, LocalId, 31);
  }
#endif
  return sycl::detail::spirv::GroupBroadcast(g, x, local_id);
#else
  (void)g;
  (void)x;
  (void)local_id;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
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
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
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
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
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
  static_assert(std::is_same_v<decltype(binary_op(x, x)), T>,
                "Result type of binary_op must match scan accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
  if constexpr (ext::oneapi::experimental::is_user_constructed_group_v<Group>) {
    return detail::masked_scan_cuda_shfls<__spv::GroupOperation::ExclusiveScan>(
        g, x, binary_op,
        sycl::detail::ExtractMask(sycl::detail::GetMask(g))[0]);
  }
#endif
  // For the first work item in the group, we cannot return the result
  // of calc when T is a signed char or short type and the
  // BinaryOperation is maximum or minimum.  calc uses SPIRV group
  // collective instructions, which only operate on 32 or 64 bit
  // integers. So, when using calc with a short or char type, the
  // argument is converted to a 32 bit integer, the 32 bit group
  // operation is performed, and then converted back to the original
  // short or char type. For an exclusive scan, the first work item
  // returns the identity for the supplied operation. However, the
  // identity of a 32 bit signed integer maximum or minimum when
  // converted to a signed char or short does not correspond to the
  // identity of a signed char or short maximum or minimum. For
  // example, the identity of a signed 32 bit maximum is
  // INT_MIN=-2**31, and when converted to a signed char, results in
  // 0. However, the identity of a signed char maximum is
  // SCHAR_MIN=-2**7. Therefore, we need the following check to
  // circumvent this issue.
  auto res = sycl::detail::calc<__spv::GroupOperation::ExclusiveScan>(
      g, typename sycl::detail::GroupOpTag<T>::type(), x, binary_op);
  if constexpr ((std::is_same_v<signed char, T> ||
                 std::is_same_v<signed short, T> ||
                 (std::is_signed_v<char> && std::is_same_v<char, T>)) &&
                detail::is_max_or_min<BinaryOperation>::value) {
    auto local_id = sycl::detail::get_local_linear_id(g);
    if (local_id == 0)
      return sycl::known_identity_v<BinaryOperation, T>;
  }
  return res;
#else
  (void)g;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
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
exclusive_scan_over_group(Group g, T x, BinaryOperation) {
#ifdef __SYCL_DEVICE_ONLY__
  T result;
  result.real(exclusive_scan_over_group(g, x.real(), sycl::plus<>()));
  result.imag(exclusive_scan_over_group(g, x.imag(), sycl::plus<>()));
  return result;
#else
  (void)g;
  (void)x;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
#endif
}

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                  detail::is_vector_arithmetic_or_complex<T>::value &&
                  detail::is_native_op<T, BinaryOperation>::value),
                 T>
exclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
  static_assert(std::is_same_v<decltype(binary_op(x, x)), T>,
                "Result type of binary_op must match scan accumulation type.");
  T result;
  typename detail::get_scalar_binary_op<BinaryOperation>::type
      scalar_binary_op{};
  for (int s = 0; s < x.size(); ++s) {
    result[s] = exclusive_scan_over_group(g, x[s], scalar_binary_op);
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
  static_assert(std::is_same_v<decltype(binary_op(init, x)), T>,
                "Result type of binary_op must match scan accumulation type.");
  T result;
  typename detail::get_scalar_binary_op<BinaryOperation>::type
      scalar_binary_op{};
  for (int s = 0; s < x.size(); ++s) {
    result[s] = exclusive_scan_over_group(g, x[s], init[s], scalar_binary_op);
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
  static_assert(std::is_same_v<decltype(binary_op(init, x)), T>,
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
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
#endif
}

// ---- joint_exclusive_scan
template <typename Group, typename InPtr, typename OutPtr, typename T,
          class BinaryOperation>
std::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer_v<InPtr> &&
     detail::is_pointer_v<OutPtr> &&
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
  static_assert(std::is_same_v<decltype(binary_op(init, *first)), T>,
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
      x = {};
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
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
#endif
}

template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation>
std::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer_v<InPtr> &&
     detail::is_pointer_v<OutPtr> &&
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
  static_assert(std::is_same_v<decltype(binary_op(*first, *first)),
                               typename detail::remove_pointer<OutPtr>::type>,
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
  static_assert(std::is_same_v<decltype(binary_op(x, x)), T>,
                "Result type of binary_op must match scan accumulation type.");
  T result;
  typename detail::get_scalar_binary_op<BinaryOperation>::type
      scalar_binary_op{};
  for (int s = 0; s < x.size(); ++s) {
    result[s] = inclusive_scan_over_group(g, x[s], scalar_binary_op);
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
  static_assert(std::is_same_v<decltype(binary_op(x, x)), T>,
                "Result type of binary_op must match scan accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)
  if constexpr (ext::oneapi::experimental::is_user_constructed_group_v<Group>) {
    return detail::masked_scan_cuda_shfls<__spv::GroupOperation::InclusiveScan>(
        g, x, binary_op,
        sycl::detail::ExtractMask(sycl::detail::GetMask(g))[0]);
  }
#endif
  return sycl::detail::calc<__spv::GroupOperation::InclusiveScan>(
      g, typename sycl::detail::GroupOpTag<T>::type(), x, binary_op);
#else
  (void)g;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
#endif
}

// complex specializaiton
template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                  detail::is_complex<T>::value &&
                  detail::is_native_op<T, sycl::plus<T>>::value &&
                  detail::is_plus<T, BinaryOperation>::value),
                 T>
inclusive_scan_over_group(Group g, T x, BinaryOperation) {
#ifdef __SYCL_DEVICE_ONLY__
  T result;
  result.real(inclusive_scan_over_group(g, x.real(), sycl::plus<>()));
  result.imag(inclusive_scan_over_group(g, x.imag(), sycl::plus<>()));
  return result;
#else
  (void)g;
  (void)x;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
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
  static_assert(std::is_same_v<decltype(binary_op(init, x)), T>,
                "Result type of binary_op must match scan accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  T y = x;
  if (sycl::detail::get_local_linear_id(g) == 0) {
    y = binary_op(init, y);
  }
  return inclusive_scan_over_group(g, y, binary_op);
#else
  (void)g;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
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
  static_assert(std::is_same_v<decltype(binary_op(init, x)), T>,
                "Result type of binary_op must match scan accumulation type.");
  T result;
  typename detail::get_scalar_binary_op<BinaryOperation>::type
      scalar_binary_op{};
  for (int s = 0; s < x.size(); ++s) {
    result[s] = inclusive_scan_over_group(g, x[s], scalar_binary_op, init[s]);
  }
  return result;
}

// ---- joint_inclusive_scan
template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation, typename T>
std::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer_v<InPtr> &&
     detail::is_pointer_v<OutPtr> &&
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
  static_assert(std::is_same_v<decltype(binary_op(init, *first)), T>,
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
      x = {};
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
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Group algorithms are not supported on host.");
#endif
}

template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation>
std::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer_v<InPtr> &&
     detail::is_pointer_v<OutPtr> &&
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
  static_assert(std::is_same_v<decltype(binary_op(*first, *first)),
                               typename detail::remove_pointer<OutPtr>::type>,
                "Result type of binary_op must match scan accumulation type.");

  using T = typename detail::remove_pointer<OutPtr>::type;
  T init = detail::identity_for_ga_op<T, BinaryOperation>();
  return joint_inclusive_scan(g, first, last, result, binary_op, init);
}

} // namespace _V1
} // namespace sycl
