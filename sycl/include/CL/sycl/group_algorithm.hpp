//==----------- group_algorithm.hpp ------------------------------------==//
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
#include <CL/sycl/detail/spirv.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/functional.hpp>
#include <CL/sycl/group.hpp>
#include <CL/sycl/known_identity.hpp>
#include <CL/sycl/nd_item.hpp>
#include <CL/sycl/sub_group.hpp>
#include <sycl/ext/oneapi/functional.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
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
detail::enable_if_t<(is_group_v<std::decay_t<Group>> &&
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
detail::enable_if_t<(is_group_v<std::decay_t<Group>> &&
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

template <typename Group, typename V, typename T, class BinaryOperation>
detail::enable_if_t<(is_group_v<std::decay_t<Group>> &&
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
detail::enable_if_t<(is_group_v<std::decay_t<Group>> &&
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

// ---- joint_reduce
template <typename Group, typename Ptr, class BinaryOperation>
detail::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer<Ptr>::value &&
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
  T partial = sycl::known_identity_v<BinaryOperation, T>;
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
    (is_group_v<std::decay_t<Group>> && detail::is_pointer<Ptr>::value &&
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
  T partial = sycl::known_identity_v<BinaryOperation, T>;
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
detail::enable_if_t<is_group_v<std::decay_t<Group>>, bool>
any_of_group(Group, bool pred) {
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
detail::enable_if_t<
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
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

// ---- all_of_group
template <typename Group>
detail::enable_if_t<is_group_v<std::decay_t<Group>>, bool>
all_of_group(Group, bool pred) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::GroupAll<Group>(pred);
#else
  (void)pred;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, class Predicate>
detail::enable_if_t<is_group_v<std::decay_t<Group>>, bool>
all_of_group(Group g, T x, Predicate pred) {
  return all_of_group(g, pred(x));
}

// ---- joint_all_of
template <typename Group, typename Ptr, class Predicate>
detail::enable_if_t<
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
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

// ---- none_of_group
template <typename Group>
detail::enable_if_t<is_group_v<std::decay_t<Group>>, bool>
none_of_group(Group, bool pred) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::GroupAll<Group>(!pred);
#else
  (void)pred;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, class Predicate>
detail::enable_if_t<is_group_v<std::decay_t<Group>>, bool>
none_of_group(Group g, T x, Predicate pred) {
  return none_of_group(g, pred(x));
}

// ---- joint_none_of
template <typename Group, typename Ptr, class Predicate>
detail::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer<Ptr>::value), bool>
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

// ---- shift_group_left
template <typename Group, typename T>
detail::enable_if_t<(std::is_same<std::decay_t<Group>, sub_group>::value &&
                     detail::is_arithmetic<T>::value),
                    T>
shift_group_left(Group, T x, typename Group::linear_id_type delta = 1) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::SubgroupShuffleDown(x, delta);
#else
  (void)x;
  (void)delta;
  throw runtime_error("Sub-groups are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

// ---- shift_group_right
template <typename Group, typename T>
detail::enable_if_t<(std::is_same<std::decay_t<Group>, sub_group>::value &&
                     detail::is_arithmetic<T>::value),
                    T>
shift_group_right(Group, T x, typename Group::linear_id_type delta = 1) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::SubgroupShuffleUp(x, delta);
#else
  (void)x;
  (void)delta;
  throw runtime_error("Sub-groups are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

// ---- permute_group_by_xor
template <typename Group, typename T>
detail::enable_if_t<(std::is_same<std::decay_t<Group>, sub_group>::value &&
                     detail::is_arithmetic<T>::value),
                    T>
permute_group_by_xor(Group, T x, typename Group::linear_id_type mask) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::SubgroupShuffleXor(x, mask);
#else
  (void)x;
  (void)mask;
  throw runtime_error("Sub-groups are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

// ---- select_from_group
template <typename Group, typename T>
detail::enable_if_t<(std::is_same<std::decay_t<Group>, sub_group>::value &&
                     detail::is_arithmetic<T>::value),
                    T>
select_from_group(Group, T x, typename Group::id_type local_id) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::SubgroupShuffle(x, local_id);
#else
  (void)x;
  (void)local_id;
  throw runtime_error("Sub-groups are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

// ---- group_broadcast
template <typename Group, typename T>
detail::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                     detail::is_scalar_arithmetic<T>::value),
                    T>
group_broadcast(Group, T x, typename Group::id_type local_id) {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::spirv::GroupBroadcast<Group>(x, local_id);
#else
  (void)x;
  (void)local_id;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T>
detail::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                     detail::is_scalar_arithmetic<T>::value),
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
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T>
detail::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                     detail::is_scalar_arithmetic<T>::value),
                    T>
group_broadcast(Group g, T x) {
#ifdef __SYCL_DEVICE_ONLY__
  return group_broadcast(g, x, 0);
#else
  (void)g;
  (void)x;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

// ---- exclusive_scan_over_group
template <typename Group, typename T, class BinaryOperation>
detail::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                     detail::is_scalar_arithmetic<T>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
exclusive_scan_over_group(Group, T x, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(std::is_same<decltype(binary_op(x, x)), T>::value ||
                    (std::is_same<T, half>::value &&
                     std::is_same<decltype(binary_op(x, x)), float>::value),
                "Result type of binary_op must match scan accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::calc<T, __spv::GroupOperation::ExclusiveScan,
                            sycl::detail::spirv::group_scope<Group>::value>(
      typename sycl::detail::GroupOpTag<T>::type(), x, binary_op);
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, class BinaryOperation>
detail::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                     detail::is_vector_arithmetic<T>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
exclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(x[0], x[0])),
                   typename T::element_type>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(x[0], x[0])), float>::value),
      "Result type of binary_op must match scan accumulation type.");
  T result;
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = exclusive_scan_over_group(g, x[s], binary_op);
  }
  return result;
}

template <typename Group, typename V, typename T, class BinaryOperation>
detail::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                     detail::is_vector_arithmetic<V>::value &&
                     detail::is_vector_arithmetic<T>::value &&
                     detail::is_native_op<V, BinaryOperation>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
exclusive_scan_over_group(Group g, V x, T init, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(init[0], x[0])),
                   typename T::element_type>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(init[0], x[0])), float>::value),
      "Result type of binary_op must match scan accumulation type.");
  T result;
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = exclusive_scan_over_group(g, x[s], init[s], binary_op);
  }
  return result;
}

template <typename Group, typename V, typename T, class BinaryOperation>
detail::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                     detail::is_scalar_arithmetic<V>::value &&
                     detail::is_scalar_arithmetic<T>::value &&
                     detail::is_native_op<V, BinaryOperation>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
exclusive_scan_over_group(Group g, V x, T init, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(std::is_same<decltype(binary_op(init, x)), T>::value ||
                    (std::is_same<T, half>::value &&
                     std::is_same<decltype(binary_op(init, x)), float>::value),
                "Result type of binary_op must match scan accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  typename Group::linear_id_type local_linear_id =
      sycl::detail::get_local_linear_id(g);
  if (local_linear_id == 0) {
    x = binary_op(init, x);
  }
  T scan = exclusive_scan_over_group(g, x, binary_op);
  if (local_linear_id == 0) {
    scan = init;
  }
  return scan;
#else
  (void)g;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

// ---- joint_exclusive_scan
template <typename Group, typename InPtr, typename OutPtr, typename T,
          class BinaryOperation>
detail::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer<InPtr>::value &&
     detail::is_pointer<OutPtr>::value &&
     detail::is_arithmetic<
         typename detail::remove_pointer<InPtr>::type>::value &&
     detail::is_arithmetic<T>::value &&
     detail::is_native_op<typename detail::remove_pointer<InPtr>::type,
                          BinaryOperation>::value &&
     detail::is_native_op<T, BinaryOperation>::value),
    OutPtr>
joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result, T init,
                     BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(*first, *first)), T>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(*first, *first)), float>::value),
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
  typename detail::remove_pointer<OutPtr>::type carry = init;
  for (ptrdiff_t chunk = 0; chunk < roundup(N, stride); chunk += stride) {
    ptrdiff_t i = chunk + offset;
    if (i < N) {
      x = first[i];
    }
    typename detail::remove_pointer<OutPtr>::type out =
        exclusive_scan_over_group(g, x, carry, binary_op);
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
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation>
detail::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer<InPtr>::value &&
     detail::is_pointer<OutPtr>::value &&
     detail::is_arithmetic<
         typename detail::remove_pointer<InPtr>::type>::value &&
     detail::is_native_op<typename detail::remove_pointer<InPtr>::type,
                          BinaryOperation>::value),
    OutPtr>
joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                     BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(*first, *first)),
                   typename detail::remove_pointer<OutPtr>::type>::value ||
          (std::is_same<typename detail::remove_pointer<OutPtr>::type,
                        half>::value &&
           std::is_same<decltype(binary_op(*first, *first)), float>::value),
      "Result type of binary_op must match scan accumulation type.");
  return joint_exclusive_scan(
      g, first, last, result,
      sycl::known_identity_v<BinaryOperation,
                             typename detail::remove_pointer<OutPtr>::type>,
      binary_op);
}

// ---- inclusive_scan_over_group
template <typename Group, typename T, class BinaryOperation>
detail::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                     detail::is_vector_arithmetic<T>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
inclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(x[0], x[0])),
                   typename T::element_type>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(x[0], x[0])), float>::value),
      "Result type of binary_op must match scan accumulation type.");
  T result;
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = inclusive_scan_over_group(g, x[s], binary_op);
  }
  return result;
}

template <typename Group, typename T, class BinaryOperation>
detail::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                     detail::is_scalar_arithmetic<T>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
inclusive_scan_over_group(Group, T x, BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(std::is_same<decltype(binary_op(x, x)), T>::value ||
                    (std::is_same<T, half>::value &&
                     std::is_same<decltype(binary_op(x, x)), float>::value),
                "Result type of binary_op must match scan accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::calc<T, __spv::GroupOperation::InclusiveScan,
                            sycl::detail::spirv::group_scope<Group>::value>(
      typename sycl::detail::GroupOpTag<T>::type(), x, binary_op);
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename V, class BinaryOperation, typename T>
detail::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                     detail::is_scalar_arithmetic<V>::value &&
                     detail::is_scalar_arithmetic<T>::value &&
                     detail::is_native_op<V, BinaryOperation>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
inclusive_scan_over_group(Group g, V x, BinaryOperation binary_op, T init) {
  // FIXME: Do not special-case for half precision
  static_assert(std::is_same<decltype(binary_op(init, x)), T>::value ||
                    (std::is_same<T, half>::value &&
                     std::is_same<decltype(binary_op(init, x)), float>::value),
                "Result type of binary_op must match scan accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  if (sycl::detail::get_local_linear_id(g) == 0) {
    x = binary_op(init, x);
  }
  return inclusive_scan_over_group(g, x, binary_op);
#else
  (void)g;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename V, class BinaryOperation, typename T>
detail::enable_if_t<(is_group_v<std::decay_t<Group>> &&
                     detail::is_vector_arithmetic<V>::value &&
                     detail::is_vector_arithmetic<T>::value &&
                     detail::is_native_op<V, BinaryOperation>::value &&
                     detail::is_native_op<T, BinaryOperation>::value),
                    T>
inclusive_scan_over_group(Group g, V x, BinaryOperation binary_op, T init) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(init[0], x[0])), T>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(init[0], x[0])), float>::value),
      "Result type of binary_op must match scan accumulation type.");
  T result;
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = inclusive_scan_over_group(g, x[s], binary_op, init[s]);
  }
  return result;
}

// ---- joint_inclusive_scan
template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation, typename T>
detail::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer<InPtr>::value &&
     detail::is_pointer<OutPtr>::value &&
     detail::is_arithmetic<
         typename detail::remove_pointer<InPtr>::type>::value &&
     detail::is_arithmetic<T>::value &&
     detail::is_native_op<typename detail::remove_pointer<InPtr>::type,
                          BinaryOperation>::value &&
     detail::is_native_op<T, BinaryOperation>::value),
    OutPtr>
joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                     BinaryOperation binary_op, T init) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(init, *first)), T>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(init, *first)), float>::value),
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
  typename detail::remove_pointer<OutPtr>::type carry = init;
  for (ptrdiff_t chunk = 0; chunk < roundup(N, stride); chunk += stride) {
    ptrdiff_t i = chunk + offset;
    if (i < N) {
      x = first[i];
    }
    typename detail::remove_pointer<OutPtr>::type out =
        inclusive_scan_over_group(g, x, binary_op, carry);
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
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation>
detail::enable_if_t<
    (is_group_v<std::decay_t<Group>> && detail::is_pointer<InPtr>::value &&
     detail::is_pointer<OutPtr>::value &&
     detail::is_arithmetic<
         typename detail::remove_pointer<InPtr>::type>::value &&
     detail::is_native_op<typename detail::remove_pointer<InPtr>::type,
                          BinaryOperation>::value),
    OutPtr>
joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                     BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(*first, *first)),
                   typename detail::remove_pointer<OutPtr>::type>::value ||
          (std::is_same<typename detail::remove_pointer<OutPtr>::type,
                        half>::value &&
           std::is_same<decltype(binary_op(*first, *first)), float>::value),
      "Result type of binary_op must match scan accumulation type.");
  return joint_inclusive_scan(
      g, first, last, result, binary_op,
      sycl::known_identity_v<BinaryOperation,
                             typename detail::remove_pointer<OutPtr>::type>);
}

namespace detail {
template <typename G> struct group_barrier_scope {};
template <> struct group_barrier_scope<sycl::sub_group> {
  constexpr static auto Scope = __spv::Scope::Subgroup;
};
template <int D> struct group_barrier_scope<sycl::group<D>> {
  constexpr static auto Scope = __spv::Scope::Workgroup;
};
} // namespace detail

template <typename Group>
typename std::enable_if<is_group_v<Group>>::type
group_barrier(Group, memory_scope FenceScope = Group::fence_scope) {
  (void)FenceScope;
#ifdef __SYCL_DEVICE_ONLY__
  // Per SYCL spec, group_barrier must perform both control barrier and memory
  // fence operations. All work-items execute a release fence prior to
  // barrier and acquire fence afterwards. The rest of semantics flags specify
  // which type of memory this behavior is applied to.
  __spirv_ControlBarrier(detail::group_barrier_scope<Group>::Scope,
                         sycl::detail::spirv::getScope(FenceScope),
                         __spv::MemorySemanticsMask::SequentiallyConsistent |
                             __spv::MemorySemanticsMask::SubgroupMemory |
                             __spv::MemorySemanticsMask::WorkgroupMemory |
                             __spv::MemorySemanticsMask::CrossWorkgroupMemory);
#else
  throw sycl::runtime_error("Barriers are not supported on host device",
                            PI_INVALID_DEVICE);
#endif
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
