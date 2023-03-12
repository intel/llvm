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
#include <sycl/detail/spirv.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/ext/oneapi/atomic.hpp>
#include <sycl/ext/oneapi/functional.hpp>
#include <sycl/ext/oneapi/sub_group.hpp>
#include <sycl/group.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/nd_item.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi {

// EnableIf shorthands for algorithms that depend only on type
template <typename T>
using EnableIfIsScalarArithmetic =
    sycl::detail::enable_if_t<sycl::detail::is_scalar_arithmetic<T>::value, T>;

template <typename T>
using EnableIfIsVectorArithmetic =
    sycl::detail::enable_if_t<sycl::detail::is_vector_arithmetic<T>::value, T>;

template <typename Ptr, typename T>
using EnableIfIsPointer =
    sycl::detail::enable_if_t<sycl::detail::is_pointer<Ptr>::value, T>;

template <typename T>
using EnableIfIsTriviallyCopyable =
    sycl::detail::enable_if_t<std::is_trivially_copyable<T>::value &&
                                  !sycl::detail::is_vector_arithmetic<T>::value,
                              T>;

// EnableIf shorthands for algorithms that depend on type and an operator
template <typename T, typename BinaryOperation>
using EnableIfIsScalarArithmeticNativeOp = sycl::detail::enable_if_t<
    sycl::detail::is_scalar_arithmetic<T>::value &&
        sycl::detail::is_native_op<T, BinaryOperation>::value,
    T>;

template <typename T, typename BinaryOperation>
using EnableIfIsVectorArithmeticNativeOp = sycl::detail::enable_if_t<
    sycl::detail::is_vector_arithmetic<T>::value &&
        sycl::detail::is_native_op<T, BinaryOperation>::value,
    T>;

// TODO: Lift TriviallyCopyable restriction eventually
template <typename T, typename BinaryOperation>
using EnableIfIsNonNativeOp = sycl::detail::enable_if_t<
    (!sycl::detail::is_scalar_arithmetic<T>::value &&
     !sycl::detail::is_vector_arithmetic<T>::value &&
     std::is_trivially_copyable<T>::value) ||
        !sycl::detail::is_native_op<T, BinaryOperation>::value,
    T>;

template <typename Group>
__SYCL2020_DEPRECATED(
    "ext::oneapi::all_of is deprecated. Use all_of_group instead.")
detail::enable_if_t<detail::is_generic_group<Group>::value, bool> all_of(
    Group g, bool pred) {
  return all_of_group(g, pred);
}

template <typename Group, typename T, class Predicate>
__SYCL2020_DEPRECATED(
    "ext::oneapi::all_of is deprecated. Use all_of_group instead.")
detail::enable_if_t<detail::is_generic_group<Group>::value, bool> all_of(
    Group g, T x, Predicate pred) {
  return all_of_group(g, pred(x));
}

template <typename Group, typename Ptr, class Predicate>
__SYCL2020_DEPRECATED(
    "ext::oneapi::all_of is deprecated. Use joint_all_of instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_pointer<Ptr>::value),
                    bool> all_of(Group g, Ptr first, Ptr last, Predicate pred) {
  return joint_all_of(g, first, last, pred);
}

template <typename Group>
__SYCL2020_DEPRECATED(
    "ext::oneapi::any_of is deprecated. Use any_of_group instead.")
detail::enable_if_t<detail::is_generic_group<Group>::value, bool> any_of(
    Group g, bool pred) {
  return any_of_group(g, pred);
}

template <typename Group, typename T, class Predicate>
__SYCL2020_DEPRECATED(
    "ext::oneapi::any_of is deprecated. Use any_of_group instead.")
detail::enable_if_t<detail::is_generic_group<Group>::value, bool> any_of(
    Group g, T x, Predicate pred) {
  return any_of_group(g, pred(x));
}

template <typename Group, typename Ptr, class Predicate>
__SYCL2020_DEPRECATED(
    "ext::oneapi::any_of is deprecated. Use joint_any_of instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_pointer<Ptr>::value),
                    bool> any_of(Group g, Ptr first, Ptr last, Predicate pred) {
  return joint_any_of(g, first, last, pred);
}

template <typename Group>
__SYCL2020_DEPRECATED(
    "ext::oneapi::none_of is deprecated. Use none_of_group instead.")
detail::enable_if_t<detail::is_generic_group<Group>::value, bool> none_of(
    Group g, bool pred) {
  return none_of_group(g, pred);
}

template <typename Group, typename T, class Predicate>
__SYCL2020_DEPRECATED(
    "ext::oneapi::none_of is deprecated. Use none_of_group instead.")
detail::enable_if_t<detail::is_generic_group<Group>::value, bool> none_of(
    Group g, T x, Predicate pred) {
  return none_of_group(g, pred(x));
}

template <typename Group, typename Ptr, class Predicate>
__SYCL2020_DEPRECATED(
    "ext::oneapi::none_of is deprecated. Use joint_none_of instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_pointer<Ptr>::value),
                    bool> none_of(Group g, Ptr first, Ptr last,
                                  Predicate pred) {
  return joint_none_of(g, first, last, pred);
}

template <typename Group, typename T>
__SYCL2020_DEPRECATED(
    "ext::oneapi::broadcast is deprecated. Use group_broadcast instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     std::is_trivially_copyable<T>::value &&
                     !detail::is_vector_arithmetic<T>::value),
                    T> broadcast(Group, T x, typename Group::id_type local_id) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::GroupBroadcast<Group>(x, local_id);
#else
  (void)x;
  (void)local_id;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename Group, typename T>
__SYCL2020_DEPRECATED(
    "ext::oneapi::broadcast is deprecated. Use group_broadcast instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_vector_arithmetic<T>::value),
                    T> broadcast(Group g, T x,
                                 typename Group::id_type local_id) {
#ifdef __SYCL_DEVICE_ONLY__
  T result;
  for (int s = 0; s < x.size(); ++s) {
    result[s] = broadcast(g, x[s], local_id);
  }
  return result;
#else
  (void)g;
  (void)x;
  (void)local_id;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename Group, typename T>
__SYCL2020_DEPRECATED(
    "ext::oneapi::broadcast is deprecated. Use group_broadcast instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     std::is_trivially_copyable<T>::value &&
                     !detail::is_vector_arithmetic<T>::value),
                    T> broadcast(Group g, T x,
                                 typename Group::linear_id_type
                                     linear_local_id) {
#ifdef __SYCL_DEVICE_ONLY__
  return broadcast(
      g, x,
      sycl::detail::linear_id_to_id(g.get_local_range(), linear_local_id));
#else
  (void)g;
  (void)x;
  (void)linear_local_id;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename Group, typename T>
__SYCL2020_DEPRECATED(
    "ext::oneapi::broadcast is deprecated. Use group_broadcast instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_vector_arithmetic<T>::value),
                    T> broadcast(Group g, T x,
                                 typename Group::linear_id_type
                                     linear_local_id) {
#ifdef __SYCL_DEVICE_ONLY__
  T result;
  for (int s = 0; s < x.size(); ++s) {
    result[s] = broadcast(g, x[s], linear_local_id);
  }
  return result;
#else
  (void)g;
  (void)x;
  (void)linear_local_id;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename Group, typename T>
__SYCL2020_DEPRECATED(
    "ext::oneapi::broadcast is deprecated. Use group_broadcast instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     std::is_trivially_copyable<T>::value &&
                     !detail::is_vector_arithmetic<T>::value),
                    T> broadcast(Group g, T x) {
#ifdef __SYCL_DEVICE_ONLY__
  return broadcast(g, x, 0);
#else
  (void)g;
  (void)x;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename Group, typename T>
__SYCL2020_DEPRECATED(
    "ext::oneapi::broadcast is deprecated. Use group_broadcast instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_vector_arithmetic<T>::value),
                    T> broadcast(Group g, T x) {
#ifdef __SYCL_DEVICE_ONLY__
  T result;
  for (int s = 0; s < x.size(); ++s) {
    result[s] = broadcast(g, x[s]);
  }
  return result;
#else
  (void)g;
  (void)x;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, class BinaryOperation>
__SYCL2020_DEPRECATED(
    "ext::oneapi::reduce is deprecated. Use reduce_over_group instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_scalar_arithmetic<T>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T> reduce(Group g, T x, BinaryOperation binary_op) {
  return reduce_over_group(g, x, binary_op);
}

template <typename Group, typename T, class BinaryOperation>
__SYCL2020_DEPRECATED(
    "ext::oneapi::reduce is deprecated. Use reduce_over_group instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_vector_arithmetic<T>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T> reduce(Group g, T x, BinaryOperation binary_op) {
  return reduce_over_group(g, x, binary_op);
}

template <typename Group, typename T, class BinaryOperation>
__SYCL2020_DEPRECATED(
    "ext::oneapi::reduce is deprecated. Use reduce_over_group instead.")
detail::enable_if_t<(detail::is_sub_group<Group>::value &&
                     std::is_trivially_copyable<T>::value &&
                     (!detail::is_arithmetic<T>::value ||
                      !detail::is_native_op<T, BinaryOperation>::value)),
                    T> reduce(Group g, T x, BinaryOperation op) {
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
__SYCL2020_DEPRECATED(
    "ext::oneapi::reduce is deprecated. Use reduce_over_group instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_scalar_arithmetic<V>::value &&
                     detail::is_scalar_arithmetic<T>::value &&
                     detail::is_native_op<V, BinaryOperation>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T> reduce(Group g, V x, T init, BinaryOperation binary_op) {
  return reduce_over_group(g, x, init, binary_op);
}

template <typename Group, typename V, typename T, class BinaryOperation>
__SYCL2020_DEPRECATED(
    "ext::oneapi::reduce is deprecated. Use reduce_over_group instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_vector_arithmetic<V>::value &&
                     detail::is_vector_arithmetic<T>::value &&
                     detail::is_native_op<V, BinaryOperation>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T> reduce(Group g, V x, T init, BinaryOperation binary_op) {
  return reduce_over_group(g, x, init, binary_op);
}

template <typename Group, typename V, typename T, class BinaryOperation>
__SYCL2020_DEPRECATED(
    "ext::oneapi::reduce is deprecated. Use reduce_over_group instead.")
detail::enable_if_t<(detail::is_sub_group<Group>::value &&
                     std::is_trivially_copyable<T>::value &&
                     std::is_trivially_copyable<V>::value &&
                     (!detail::is_arithmetic<T>::value ||
                      !detail::is_arithmetic<V>::value ||
                      !detail::is_native_op<T, BinaryOperation>::value)),
                    T> reduce(Group g, V x, T init, BinaryOperation op) {
  T result = x;
  for (int mask = 1; mask < g.get_max_local_range()[0]; mask *= 2) {
    T tmp = g.shuffle_xor(result, id<1>(mask));
    if ((g.get_local_id()[0] ^ mask) < g.get_local_range()[0]) {
      result = op(result, tmp);
    }
  }
  return g.shuffle(op(init, result), 0);
}

template <typename Group, typename Ptr, class BinaryOperation>
__SYCL2020_DEPRECATED(
    "ext::oneapi::reduce is deprecated. Use joint_reduce instead.")
detail::enable_if_t<
    (detail::is_generic_group<Group>::value && detail::is_pointer<Ptr>::value &&
     detail::is_arithmetic<typename detail::remove_pointer<Ptr>::type>::value),
    typename detail::remove_pointer<Ptr>::type> reduce(Group g, Ptr first,
                                                       Ptr last,
                                                       BinaryOperation
                                                           binary_op) {
  return joint_reduce(g, first, last, binary_op);
}

template <typename Group, typename Ptr, typename T, class BinaryOperation>
__SYCL2020_DEPRECATED(
    "ext::oneapi::reduce is deprecated. Use joint_reduce instead.")
detail::enable_if_t<
    (detail::is_generic_group<Group>::value && detail::is_pointer<Ptr>::value &&
     detail::is_arithmetic<typename detail::remove_pointer<Ptr>::type>::value &&
     detail::is_arithmetic<T>::value &&
     detail::is_native_op<typename detail::remove_pointer<Ptr>::type,
                          BinaryOperation>::value &&
     detail::is_native_op<T, BinaryOperation>::value),
    T> reduce(Group g, Ptr first, Ptr last, T init, BinaryOperation binary_op) {
  return joint_reduce(g, first, last, init, binary_op);
}

template <typename Group, typename T, class BinaryOperation>
__SYCL2020_DEPRECATED("ext::oneapi::exclusive_scan is deprecated. Use "
                      "exclusive_scan_over_group instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_scalar_arithmetic<T>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T> exclusive_scan(Group g, T x, BinaryOperation binary_op) {
  return exclusive_scan_over_group(g, x, binary_op);
}

template <typename Group, typename T, class BinaryOperation>
__SYCL2020_DEPRECATED("ext::oneapi::exclusive_scan is deprecated. Use "
                      "exclusive_scan_over_group instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_vector_arithmetic<T>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T> exclusive_scan(Group g, T x, BinaryOperation binary_op) {
  return exclusive_scan_over_group(g, x, binary_op);
}

template <typename Group, typename V, typename T, class BinaryOperation>
__SYCL2020_DEPRECATED("ext::oneapi::exclusive_scan is deprecated. Use "
                      "exclusive_scan_over_group instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_vector_arithmetic<V>::value &&
                     detail::is_vector_arithmetic<T>::value &&
                     detail::is_native_op<V, BinaryOperation>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T> exclusive_scan(Group g, V x, T init,
                                      BinaryOperation binary_op) {
  return exclusive_scan_over_group(g, x, init, binary_op);
}

template <typename Group, typename V, typename T, class BinaryOperation>
__SYCL2020_DEPRECATED("ext::oneapi::exclusive_scan is deprecated. Use "
                      "exclusive_scan_over_group instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_scalar_arithmetic<V>::value &&
                     detail::is_scalar_arithmetic<T>::value &&
                     detail::is_native_op<V, BinaryOperation>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T> exclusive_scan(Group g, V x, T init,
                                      BinaryOperation binary_op) {
  return exclusive_scan_over_group(g, x, init, binary_op);
}

template <typename Group, typename InPtr, typename OutPtr, typename T,
          class BinaryOperation>
__SYCL2020_DEPRECATED("ext::oneapi::exclusive_scan is deprecated. Use "
                      "joint_exclusive_scan instead.")
detail::enable_if_t<
    (detail::is_generic_group<Group>::value &&
     detail::is_pointer<InPtr>::value && detail::is_pointer<OutPtr>::value &&
     detail::is_arithmetic<
         typename detail::remove_pointer<InPtr>::type>::value &&
     detail::is_arithmetic<T>::value &&
     detail::is_native_op<typename detail::remove_pointer<InPtr>::type,
                          BinaryOperation>::value &&
     detail::is_native_op<T, BinaryOperation>::value),
    OutPtr> exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                           T init, BinaryOperation binary_op) {
  return joint_exclusive_scan(g, first, last, result, init, binary_op);
}

template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation>
__SYCL2020_DEPRECATED("ext::oneapi::exclusive_scan is deprecated. Use "
                      "joint_exclusive_scan instead.")
detail::enable_if_t<
    (detail::is_generic_group<Group>::value &&
     detail::is_pointer<InPtr>::value && detail::is_pointer<OutPtr>::value &&
     detail::is_arithmetic<
         typename detail::remove_pointer<InPtr>::type>::value &&
     detail::is_native_op<typename detail::remove_pointer<InPtr>::type,
                          BinaryOperation>::value),
    OutPtr> exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                           BinaryOperation binary_op) {
  return joint_exclusive_scan(g, first, last, result, binary_op);
}

template <typename Group, typename T, class BinaryOperation>
__SYCL2020_DEPRECATED("ext::oneapi::inclusive_scan is deprecated. Use "
                      "inclusive_scan_over_group instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_vector_arithmetic<T>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T> inclusive_scan(Group g, T x, BinaryOperation binary_op) {
  return inclusive_scan_over_group(g, x, binary_op);
}

template <typename Group, typename T, class BinaryOperation>
__SYCL2020_DEPRECATED("ext::oneapi::inclusive_scan is deprecated. Use "
                      "inclusive_scan_over_group instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_scalar_arithmetic<T>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T> inclusive_scan(Group g, T x, BinaryOperation binary_op) {
  return inclusive_scan_over_group(g, x, binary_op);
}

template <typename Group, typename V, class BinaryOperation, typename T>
__SYCL2020_DEPRECATED("ext::oneapi::inclusive_scan is deprecated. Use "
                      "inclusive_scan_over_group instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_scalar_arithmetic<V>::value &&
                     detail::is_scalar_arithmetic<T>::value &&
                     detail::is_native_op<V, BinaryOperation>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T> inclusive_scan(Group g, V x, BinaryOperation binary_op,
                                      T init) {
  return inclusive_scan_over_group(g, x, binary_op, init);
}

template <typename Group, typename V, class BinaryOperation, typename T>
__SYCL2020_DEPRECATED("ext::oneapi::inclusive_scan is deprecated. Use "
                      "inclusive_scan_over_group instead.")
detail::enable_if_t<(detail::is_generic_group<Group>::value &&
                     detail::is_vector_arithmetic<V>::value &&
                     detail::is_vector_arithmetic<T>::value &&
                     detail::is_native_op<V, BinaryOperation>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T> inclusive_scan(Group g, V x, BinaryOperation binary_op,
                                      T init) {
  return inclusive_scan_over_group(g, x, binary_op, init);
}

template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation, typename T>
__SYCL2020_DEPRECATED("ext::oneapi::inclusive_scan is deprecated. Use "
                      "joint_inclusive_scan instead.")
detail::enable_if_t<
    (detail::is_generic_group<Group>::value &&
     detail::is_pointer<InPtr>::value && detail::is_pointer<OutPtr>::value &&
     detail::is_arithmetic<
         typename detail::remove_pointer<InPtr>::type>::value &&
     detail::is_arithmetic<T>::value &&
     detail::is_native_op<typename detail::remove_pointer<InPtr>::type,
                          BinaryOperation>::value &&
     detail::is_native_op<T, BinaryOperation>::value),
    OutPtr> inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                           BinaryOperation binary_op, T init) {
  return joint_inclusive_scan(g, first, last, result, binary_op, init);
}

template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation>
__SYCL2020_DEPRECATED("ext::oneapi::inclusive_scan is deprecated. Use "
                      "joint_inclusive_scan instead.")
detail::enable_if_t<
    (detail::is_generic_group<Group>::value &&
     detail::is_pointer<InPtr>::value && detail::is_pointer<OutPtr>::value &&
     detail::is_arithmetic<
         typename detail::remove_pointer<InPtr>::type>::value &&
     detail::is_native_op<typename detail::remove_pointer<InPtr>::type,
                          BinaryOperation>::value),
    OutPtr> inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                           BinaryOperation binary_op) {
  return joint_inclusive_scan(g, first, last, result, binary_op);
}

template <typename Group>
__SYCL2020_DEPRECATED(
    "ext::oneapi::leader free function is deprecated. Use member function "
    "leader of the sycl::group/sycl::sub_group instead.")
detail::enable_if_t<detail::is_generic_group<Group>::value, bool> leader(
    Group g) {
#ifdef __SYCL_DEVICE_ONLY__
  typename Group::linear_id_type linear_id =
      sycl::detail::get_local_linear_id(g);
  return (linear_id == 0);
#else
  (void)g;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_ERROR_INVALID_DEVICE);
#endif
}

} // namespace ext::oneapi

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
