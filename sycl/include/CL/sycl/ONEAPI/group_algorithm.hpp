//==----------- group_algorithm.hpp --- SYCL group algorithm----------------==//
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
#include <CL/sycl/ONEAPI/sub_group.hpp>
#include <CL/sycl/detail/spirv.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/group.hpp>
#include <CL/sycl/nd_item.hpp>
#include <CL/sycl/sub_group_algorithm.hpp>

#ifndef __DISABLE_SYCL_ONEAPI_GROUP_ALGORITHMS__
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ONEAPI {

// EnableIf shorthands for algorithms that depend only on type
template <typename T>
using EnableIfIsScalarArithmetic = cl::sycl::detail::enable_if_t<
    cl::sycl::detail::is_scalar_arithmetic<T>::value, T>;

template <typename T>
using EnableIfIsVectorArithmetic = cl::sycl::detail::enable_if_t<
    cl::sycl::detail::is_vector_arithmetic<T>::value, T>;

template <typename Ptr, typename T>
using EnableIfIsPointer =
    cl::sycl::detail::enable_if_t<cl::sycl::detail::is_pointer<Ptr>::value, T>;

template <typename T>
using EnableIfIsTriviallyCopyable = cl::sycl::detail::enable_if_t<
    std::is_trivially_copyable<T>::value &&
        !cl::sycl::detail::is_vector_arithmetic<T>::value,
    T>;

// EnableIf shorthands for algorithms that depend on type and an operator
template <typename T, typename BinaryOperation>
using EnableIfIsScalarArithmeticNativeOp = cl::sycl::detail::enable_if_t<
    cl::sycl::detail::is_scalar_arithmetic<T>::value &&
        cl::sycl::detail::is_native_op<T, BinaryOperation>::value,
    T>;

template <typename T, typename BinaryOperation>
using EnableIfIsVectorArithmeticNativeOp = cl::sycl::detail::enable_if_t<
    cl::sycl::detail::is_vector_arithmetic<T>::value &&
        cl::sycl::detail::is_native_op<T, BinaryOperation>::value,
    T>;

// TODO: Lift TriviallyCopyable restriction eventually
template <typename T, typename BinaryOperation>
using EnableIfIsNonNativeOp = cl::sycl::detail::enable_if_t<
    (!cl::sycl::detail::is_scalar_arithmetic<T>::value &&
     !cl::sycl::detail::is_vector_arithmetic<T>::value &&
     std::is_trivially_copyable<T>::value) ||
        !cl::sycl::detail::is_native_op<T, BinaryOperation>::value,
    T>;

template <typename Group>
detail::enable_if_t<detail::is_generic_group<Group>::value, bool>
__SYCL2020_DEPRECATED("ONEAPI::all_of is deprecated. Use all_of_group instead.")
    all_of(Group g, bool pred) {
  return all_of_group(g, pred);
}

template <typename Group, typename T, class Predicate>
detail::enable_if_t<detail::is_generic_group<Group>::value, bool>
__SYCL2020_DEPRECATED("ONEAPI::all_of is deprecated. Use all_of_group instead.")
    all_of(Group g, T x, Predicate pred) {
  return all_of_group(g, pred(x));
}

template <typename Group, typename Ptr, class Predicate>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_pointer<Ptr>::value),
                    bool>
__SYCL2020_DEPRECATED("ONEAPI::all_of is deprecated. Use joint_all_of instead.")
    all_of(Group g, Ptr first, Ptr last, Predicate pred) {
  return joint_all_of(g, first, last, pred);
}

template <typename Group>
detail::enable_if_t<detail::is_generic_group<Group>::value, bool>
__SYCL2020_DEPRECATED("ONEAPI::any_of is deprecated. Use any_of_group instead.")
    any_of(Group g, bool pred) {
  return any_of_group(g, pred);
}

template <typename Group, typename T, class Predicate>
detail::enable_if_t<detail::is_generic_group<Group>::value, bool>
__SYCL2020_DEPRECATED("ONEAPI::any_of is deprecated. Use any_of_group instead.")
    any_of(Group g, T x, Predicate pred) {
  return any_of_group(g, pred(x));
}

template <typename Group, typename Ptr, class Predicate>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_pointer<Ptr>::value),
                    bool>
__SYCL2020_DEPRECATED("ONEAPI::any_of is deprecated. Use joint_any_of instead.")
    any_of(Group g, Ptr first, Ptr last, Predicate pred) {
  return joint_any_of(g, first, last, pred);
}

template <typename Group>
detail::enable_if_t<detail::is_generic_group<Group>::value, bool>
__SYCL2020_DEPRECATED(
    "ONEAPI::none_of is deprecated. Use none_of_group instead.")
    none_of(Group g, bool pred) {
  return none_of_group(g, pred);
}

template <typename Group, typename T, class Predicate>
detail::enable_if_t<detail::is_generic_group<Group>::value, bool>
__SYCL2020_DEPRECATED(
    "ONEAPI::none_of is deprecated. Use none_of_group instead.")
    none_of(Group g, T x, Predicate pred) {
  return none_of_group(g, pred(x));
}

template <typename Group, typename Ptr, class Predicate>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_pointer<Ptr>::value),
                    bool>
__SYCL2020_DEPRECATED(
    "ONEAPI::none_of is deprecated. Use none_of_group instead.")
    none_of(Group g, Ptr first, Ptr last, Predicate pred) {
  return joint_none_of(g, first, last, pred);
}

template <typename Group, typename T>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     std::is_trivially_copyable<T>::value &&
                     !detail::is_vector_arithmetic<T>::value),
                    T>
__SYCL2020_DEPRECATED(
    "ONEAPI::broadcast is deprecated. Use group_broadcast instead.")
    broadcast(Group g, T x, typename Group::id_type local_id) {
  return group_broadcast(g, x, local_id);
}

template <typename Group, typename T>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_vector_arithmetic<T>::value),
                    T>
__SYCL2020_DEPRECATED(
    "ONEAPI::broadcast is deprecated. Use group_broadcast instead.")
    broadcast(Group g, T x, typename Group::id_type local_id) {
  return group_broadcast(g, x, local_id);
}

template <typename Group, typename T>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     std::is_trivially_copyable<T>::value &&
                     !detail::is_vector_arithmetic<T>::value),
                    T>
__SYCL2020_DEPRECATED(
    "ONEAPI::broadcast is deprecated. Use group_broadcast instead.")
    broadcast(Group g, T x, typename Group::linear_id_type linear_local_id) {
  return group_broadcast(g, x, linear_local_id);
}

template <typename Group, typename T>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_vector_arithmetic<T>::value),
                    T>
__SYCL2020_DEPRECATED(
    "ONEAPI::broadcast is deprecated. Use group_broadcast instead.")
    broadcast(Group g, T x, typename Group::linear_id_type linear_local_id) {
  return group_broadcast(g, x, linear_local_id);
}

template <typename Group, typename T>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     std::is_trivially_copyable<T>::value &&
                     !detail::is_vector_arithmetic<T>::value),
                    T>
__SYCL2020_DEPRECATED(
    "ONEAPI::broadcast is deprecated. Use group_broadcast instead.")
    broadcast(Group g, T x) {
  return group_broadcast(g, x);
}

template <typename Group, typename T>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_vector_arithmetic<T>::value),
                    T>
__SYCL2020_DEPRECATED(
    "ONEAPI::broadcast is deprecated. Use group_broadcast instead.")
    broadcast(Group g, T x) {
  return group_broadcast(g, x);
}

template <typename Group, typename T, class BinaryOperation>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_scalar_arithmetic<T>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
__SYCL2020_DEPRECATED("Thank you Mario! But our princess is in another castle!")
    reduce(Group g, T x, BinaryOperation binary_op) {
  return reduce_over_group(g, x, binary_op);
}

template <typename Group, typename T, class BinaryOperation>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_vector_arithmetic<T>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
__SYCL2020_DEPRECATED("Thank you Mario! But our princess is in another castle!")
    reduce(Group g, T x, BinaryOperation binary_op) {
  return reduce_over_group(g, x, binary_op);
}

template <typename Group, typename T, class BinaryOperation>
detail::enable_if_t<(detail::is_sub_group<Group>::value &&
                     std::is_trivially_copyable<T>::value &&
                     (!detail::is_arithmetic<T>::value ||
                      !detail::is_native_op<T, BinaryOperation>::value)),
                    T>
__SYCL2020_DEPRECATED("Thank you Mario! But our princess is in another castle!")
    reduce(Group g, T x, BinaryOperation op) {
  return reduce_over_group(g, x, op);
}

template <typename Group, typename V, typename T, class BinaryOperation>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_scalar_arithmetic<V>::value &&
                     detail::is_scalar_arithmetic<T>::value &&
                     detail::is_native_op<V, BinaryOperation>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
__SYCL2020_DEPRECATED("Thank you Mario! But our princess is in another castle!")
    reduce(Group g, V x, T init, BinaryOperation binary_op) {
  return reduce_over_group(g, x, init, binary_op);
}

template <typename Group, typename V, typename T, class BinaryOperation>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_vector_arithmetic<V>::value &&
                     detail::is_vector_arithmetic<T>::value &&
                     detail::is_native_op<V, BinaryOperation>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
__SYCL2020_DEPRECATED("Thank you Mario! But our princess is in another castle!")
    reduce(Group g, V x, T init, BinaryOperation binary_op) {
  return reduce_over_group(g, x, init, binary_op);
}

template <typename Group, typename V, typename T, class BinaryOperation>
detail::enable_if_t<(detail::is_sub_group<Group>::value &&
                     std::is_trivially_copyable<T>::value &&
                     std::is_trivially_copyable<V>::value &&
                     (!detail::is_arithmetic<T>::value ||
                      !detail::is_arithmetic<V>::value ||
                      !detail::is_native_op<T, BinaryOperation>::value)),
                    T>
__SYCL2020_DEPRECATED("Thank you Mario! But our princess is in another castle!")
    reduce(Group g, V x, T init, BinaryOperation op) {
  return reduce_over_group(g, x, init, op);
}

template <typename Group, typename Ptr, class BinaryOperation>
detail::enable_if_t<
    (detail::is_generic_group<Group>::value && detail::is_pointer<Ptr>::value &&
     detail::is_arithmetic<typename detail::remove_pointer<Ptr>::type>::value),
    typename detail::remove_pointer<Ptr>::type>
__SYCL2020_DEPRECATED("ONEAPI::reduce is deprecated. Use joint_reduce instead.")
    reduce(Group g, Ptr first, Ptr last, BinaryOperation binary_op) {
  return joint_reduce(g, first, last, binary_op);
}

template <typename Group, typename Ptr, typename T, class BinaryOperation>
detail::enable_if_t<
    (detail::is_generic_group<Group>::value && detail::is_pointer<Ptr>::value &&
     detail::is_arithmetic<typename detail::remove_pointer<Ptr>::type>::value &&
     detail::is_arithmetic<T>::value &&
     detail::is_native_op<typename detail::remove_pointer<Ptr>::type,
                          BinaryOperation>::value &&
     detail::is_native_op<T, BinaryOperation>::value),
    T>
__SYCL2020_DEPRECATED("ONEAPI::reduce is deprecated. Use joint_reduce instead.")
    reduce(Group g, Ptr first, Ptr last, T init, BinaryOperation binary_op) {
  return joint_reduce(g, first, last, init, binary_op);
}

template <typename Group, typename T, class BinaryOperation>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_scalar_arithmetic<T>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
__SYCL2020_DEPRECATED("ONEAPI::exclusive_scan is deprecated. Use "
                      "exclusive_scan_over_group instead.")
    exclusive_scan(Group g, T x, BinaryOperation binary_op) {
  return exclusive_scan_over_group(g, x, binary_op);
}

template <typename Group, typename T, class BinaryOperation>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_vector_arithmetic<T>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
__SYCL2020_DEPRECATED("ONEAPI::exclusive_scan is deprecated. Use "
                      "exclusive_scan_over_group instead.")
    exclusive_scan(Group g, T x, BinaryOperation binary_op) {
  return exclusive_scan_over_group(g, x, binary_op);
}

template <typename Group, typename V, typename T, class BinaryOperation>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_vector_arithmetic<V>::value &&
                     detail::is_vector_arithmetic<T>::value &&
                     detail::is_native_op<V, BinaryOperation>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
__SYCL2020_DEPRECATED("ONEAPI::exclusive_scan is deprecated. Use "
                      "exclusive_scan_over_group instead.")
    exclusive_scan(Group g, V x, T init, BinaryOperation binary_op) {
  return exclusive_scan_over_group(g, x, init, binary_op);
}

template <typename Group, typename V, typename T, class BinaryOperation>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_scalar_arithmetic<V>::value &&
                     detail::is_scalar_arithmetic<T>::value &&
                     detail::is_native_op<V, BinaryOperation>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
__SYCL2020_DEPRECATED("ONEAPI::exclusive_scan is deprecated. Use "
                      "exclusive_scan_over_group instead.")
    exclusive_scan(Group g, V x, T init, BinaryOperation binary_op) {
  return exclusive_scan_over_group(g, x, init, binary_op);
}

template <typename Group, typename InPtr, typename OutPtr, typename T,
          class BinaryOperation>
detail::enable_if_t<
    (detail::is_generic_group<Group>::value &&
     detail::is_pointer<InPtr>::value && detail::is_pointer<OutPtr>::value &&
     detail::is_arithmetic<
         typename detail::remove_pointer<InPtr>::type>::value &&
     detail::is_arithmetic<T>::value &&
     detail::is_native_op<typename detail::remove_pointer<InPtr>::type,
                          BinaryOperation>::value &&
     detail::is_native_op<T, BinaryOperation>::value),
    OutPtr>
__SYCL2020_DEPRECATED(
    "ONEAPI::exclusive_scan is deprecated. Use joint_exclusive_scan instead.")
    exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result, T init,
                   BinaryOperation binary_op) {
  return joint_exclusive_scan(g, first, last, result, init, binary_op);
}

template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation>
detail::enable_if_t<
    (detail::is_generic_group<Group>::value &&
     detail::is_pointer<InPtr>::value && detail::is_pointer<OutPtr>::value &&
     detail::is_arithmetic<
         typename detail::remove_pointer<InPtr>::type>::value &&
     detail::is_native_op<typename detail::remove_pointer<InPtr>::type,
                          BinaryOperation>::value),
    OutPtr>
__SYCL2020_DEPRECATED(
    "ONEAPI::exclusive_scan is deprecated. Use joint_exclusive_scan instead.")
    exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                   BinaryOperation binary_op) {
  return joint_exclusive_scan(g, first, last, result, binary_op);
}

template <typename Group, typename T, class BinaryOperation>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_vector_arithmetic<T>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
__SYCL2020_DEPRECATED("ONEAPI::inclusive_scan is deprecated. Use "
                      "inclusive_scan_over_group instead.")
    inclusive_scan(Group g, T x, BinaryOperation binary_op) {
  return inclusive_scan_over_group(g, x, binary_op);
}

template <typename Group, typename T, class BinaryOperation>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_scalar_arithmetic<T>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
__SYCL2020_DEPRECATED("ONEAPI::inclusive_scan is deprecated. Use "
                      "inclusive_scan_over_group instead.")
    inclusive_scan(Group g, T x, BinaryOperation binary_op) {
  return inclusive_scan_over_group(g, x, binary_op);
}

template <typename Group, typename V, class BinaryOperation, typename T>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_scalar_arithmetic<V>::value &&
                     detail::is_scalar_arithmetic<T>::value &&
                     detail::is_native_op<V, BinaryOperation>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
__SYCL2020_DEPRECATED("ONEAPI::inclusive_scan is deprecated. Use "
                      "inclusive_scan_over_group instead.")
    inclusive_scan(Group g, V x, BinaryOperation binary_op, T init) {
  return inclusive_scan_over_group(g, x, binary_op, init);
}

template <typename Group, typename V, class BinaryOperation, typename T>
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_vector_arithmetic<V>::value &&
                     detail::is_vector_arithmetic<T>::value &&
                     detail::is_native_op<V, BinaryOperation>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
__SYCL2020_DEPRECATED("ONEAPI::inclusive_scan is deprecated. Use "
                      "inclusive_scan_over_group instead.")
    inclusive_scan(Group g, V x, BinaryOperation binary_op, T init) {
  return inclusive_scan_over_group(g, x, binary_op, init);
}

template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation, typename T>
detail::enable_if_t<
    (detail::is_generic_group<Group>::value &&
     detail::is_pointer<InPtr>::value && detail::is_pointer<OutPtr>::value &&
     detail::is_arithmetic<
         typename detail::remove_pointer<InPtr>::type>::value &&
     detail::is_arithmetic<T>::value &&
     detail::is_native_op<typename detail::remove_pointer<InPtr>::type,
                          BinaryOperation>::value &&
     detail::is_native_op<T, BinaryOperation>::value),
    OutPtr>
__SYCL2020_DEPRECATED(
    "ONEAPI::inclusive_scan is deprecated. Use joint_inclusive_scan instead.")
    inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                   BinaryOperation binary_op, T init) {
  return joint_inclusive_scan(g, first, last, result, binary_op, init);
}

template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation>
detail::enable_if_t<
    (detail::is_generic_group<Group>::value &&
     detail::is_pointer<InPtr>::value && detail::is_pointer<OutPtr>::value &&
     detail::is_arithmetic<
         typename detail::remove_pointer<InPtr>::type>::value &&
     detail::is_native_op<typename detail::remove_pointer<InPtr>::type,
                          BinaryOperation>::value),
    OutPtr>
__SYCL2020_DEPRECATED(
    "ONEAPI::inclusive_scan is deprecated. Use joint_inclusive_scan instead.")
    inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                   BinaryOperation binary_op) {
  return joint_inclusive_scan(g, first, last, result, binary_op);
}

template <typename Group>
detail::enable_if_t<detail::is_generic_group<Group>::value, bool>
leader(Group g) {
#ifdef __SYCL_DEVICE_ONLY__
  typename Group::linear_id_type linear_id =
      sycl::detail::get_local_linear_id(g);
  return (linear_id == 0);
#else
  (void)g;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

} // namespace ONEAPI
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
#endif // __DISABLE_SYCL_ONEAPI_GROUP_ALGORITHMS__
