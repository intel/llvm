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
#include <CL/sycl/detail/spirv.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/group.hpp>
#include <CL/sycl/intel/functional.hpp>
#include <CL/sycl/intel/sub_group.hpp>

#ifndef __DISABLE_SYCL_INTEL_GROUP_ALGORITHMS__
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

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
inline size_t get_local_linear_range<intel::sub_group>(intel::sub_group g) {
  return g.get_local_range()[0];
}

template <typename Group>
typename Group::linear_id_type get_local_linear_id(Group g);

#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_GROUP_GET_LOCAL_LINEAR_ID(D)                                    \
  template <>                                                                  \
  group<D>::linear_id_type get_local_linear_id<group<D>>(group<D> g) {         \
    nd_item<D> it = cl::sycl::detail::Builder::getNDItem<D>();                 \
    return it.get_local_linear_id();                                           \
  }
__SYCL_GROUP_GET_LOCAL_LINEAR_ID(1);
__SYCL_GROUP_GET_LOCAL_LINEAR_ID(2);
__SYCL_GROUP_GET_LOCAL_LINEAR_ID(3);
#undef __SYCL_GROUP_GET_LOCAL_LINEAR_ID
#endif // __SYCL_DEVICE_ONLY__

template <>
inline intel::sub_group::linear_id_type
get_local_linear_id<intel::sub_group>(intel::sub_group g) {
  return g.get_local_id()[0];
}

template <int Dimensions>
id<Dimensions> linear_id_to_id(range<Dimensions>, size_t linear_id);
template <> inline id<1> linear_id_to_id(range<1> r, size_t linear_id) {
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

template <typename T> struct is_group : std::false_type {};

template <int Dimensions>
struct is_group<group<Dimensions>> : std::true_type {};

template <typename T> struct is_sub_group : std::false_type {};

template <> struct is_sub_group<intel::sub_group> : std::true_type {};

template <typename T>
struct is_generic_group
    : std::integral_constant<bool,
                             is_group<T>::value || is_sub_group<T>::value> {};

template <typename T, class BinaryOperation> struct identity {};

template <typename T, typename V> struct identity<T, intel::plus<V>> {
  static constexpr T value = 0;
};

template <typename T, typename V> struct identity<T, intel::minimum<V>> {
  static constexpr T value = (std::numeric_limits<T>::max)();
};

template <typename T, typename V> struct identity<T, intel::maximum<V>> {
  static constexpr T value = std::numeric_limits<T>::lowest();
};

template <typename Group, typename Ptr, class Function>
Function for_each(Group g, Ptr first, Ptr last, Function f) {
#ifdef __SYCL_DEVICE_ONLY__
  ptrdiff_t offset = detail::get_local_linear_id(g);
  ptrdiff_t stride = detail::get_local_linear_range(g);
  for (Ptr p = first + offset; p < last; p += stride) {
    f(*p);
  }
  return f;
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

} // namespace detail

namespace intel {

template <typename T>
using EnableIfIsScalarArithmetic = cl::sycl::detail::enable_if_t<
    cl::sycl::detail::is_scalar_arithmetic<T>::value, T>;

template <typename T>
using EnableIfIsVectorArithmetic = cl::sycl::detail::enable_if_t<
    cl::sycl::detail::is_vector_arithmetic<T>::value, T>;

template <typename Ptr, typename T>
using EnableIfIsPointer =
    cl::sycl::detail::enable_if_t<cl::sycl::detail::is_pointer<Ptr>::value, T>;

template <typename Group> bool all_of(Group g, bool pred) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
#ifdef __SYCL_DEVICE_ONLY__
  return detail::spirv::GroupAll<Group>(pred);
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, class Predicate>
bool all_of(Group g, T x, Predicate pred) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  return all_of(g, pred(x));
}

template <typename Group, typename Ptr, class Predicate>
EnableIfIsPointer<Ptr, bool> all_of(Group g, Ptr first, Ptr last,
                                    Predicate pred) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
#ifdef __SYCL_DEVICE_ONLY__
  bool partial = true;
  detail::for_each(g, first, last, [&](const typename Ptr::element_type &x) {
    partial &= pred(x);
  });
  return all_of(g, partial);
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group> bool any_of(Group g, bool pred) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
#ifdef __SYCL_DEVICE_ONLY__
  return detail::spirv::GroupAny<Group>(pred);
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, class Predicate>
bool any_of(Group g, T x, Predicate pred) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  return any_of(g, pred(x));
}

template <typename Group, typename Ptr, class Predicate>
EnableIfIsPointer<Ptr, bool> any_of(Group g, Ptr first, Ptr last,
                                    Predicate pred) {
#ifdef __SYCL_DEVICE_ONLY__
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  bool partial = false;
  detail::for_each(g, first, last, [&](const typename Ptr::element_type &x) {
    partial |= pred(x);
  });
  return any_of(g, partial);
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group> bool none_of(Group g, bool pred) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
#ifdef __SYCL_DEVICE_ONLY__
  return detail::spirv::GroupAll<Group>(not pred);
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, class Predicate>
bool none_of(Group g, T x, Predicate pred) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  return none_of(g, pred(x));
}

template <typename Group, typename Ptr, class Predicate>
EnableIfIsPointer<Ptr, bool> none_of(Group g, Ptr first, Ptr last,
                                     Predicate pred) {
#ifdef __SYCL_DEVICE_ONLY__
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  return not any_of(g, first, last, pred);
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T>
EnableIfIsScalarArithmetic<T> broadcast(Group g, T x,
                                        typename Group::id_type local_id) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
#ifdef __SYCL_DEVICE_ONLY__
  return detail::spirv::GroupBroadcast<Group>(x, local_id);
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T>
EnableIfIsVectorArithmetic<T> broadcast(Group g, T x,
                                        typename Group::id_type local_id) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
#ifdef __SYCL_DEVICE_ONLY__
  T result;
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = broadcast(g, x[s], local_id);
  }
  return result;
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T>
EnableIfIsScalarArithmetic<T>
broadcast(Group g, T x, typename Group::linear_id_type linear_local_id) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
#ifdef __SYCL_DEVICE_ONLY__
  return broadcast(
      g, x, detail::linear_id_to_id(g.get_local_range(), linear_local_id));
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T>
EnableIfIsVectorArithmetic<T>
broadcast(Group g, T x, typename Group::linear_id_type linear_local_id) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
#ifdef __SYCL_DEVICE_ONLY__
  T result;
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = broadcast(g, x[s], linear_local_id);
  }
  return result;
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T>
EnableIfIsScalarArithmetic<T> broadcast(Group g, T x) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
#ifdef __SYCL_DEVICE_ONLY__
  return broadcast(g, x, 0);
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T>
EnableIfIsVectorArithmetic<T> broadcast(Group g, T x) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
#ifdef __SYCL_DEVICE_ONLY__
  T result;
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = broadcast(g, x[s]);
  }
  return result;
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, class BinaryOperation>
EnableIfIsScalarArithmetic<T> reduce(Group g, T x, BinaryOperation binary_op) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(x, x)), T>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(x, x)), float>::value),
      "Result type of binary_op must match reduction accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  return detail::calc<T, __spv::GroupOperation::Reduce,
                      detail::spirv::group_scope<Group>::value>(
      typename detail::GroupOpTag<T>::type(), x, binary_op);
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, class BinaryOperation>
EnableIfIsVectorArithmetic<T> reduce(Group g, T x, BinaryOperation binary_op) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(x[0], x[0])),
                   typename T::element_type>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(x[0], x[0])), float>::value),
      "Result type of binary_op must match reduction accumulation type.");
  T result;
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = reduce(g, x[s], binary_op);
  }
  return result;
}

template <typename Group, typename V, typename T, class BinaryOperation>
EnableIfIsScalarArithmetic<T> reduce(Group g, V x, T init,
                                     BinaryOperation binary_op) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(init, x)), T>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(init, x)), float>::value),
      "Result type of binary_op must match reduction accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  return binary_op(init, reduce(g, x, binary_op));
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename V, typename T, class BinaryOperation>
EnableIfIsVectorArithmetic<T> reduce(Group g, V x, T init,
                                     BinaryOperation binary_op) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
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
    result[s] = binary_op(init[s], reduce(g, x[s], binary_op));
  }
  return result;
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename Ptr, class BinaryOperation>
EnableIfIsPointer<Ptr, typename Ptr::element_type>
reduce(Group g, Ptr first, Ptr last, BinaryOperation binary_op) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(*first, *first)),
                   typename Ptr::element_type>::value ||
          (std::is_same<typename Ptr::element_type, half>::value &&
           std::is_same<decltype(binary_op(*first, *first)), float>::value),
      "Result type of binary_op must match reduction accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  typename Ptr::element_type partial =
      detail::identity<typename Ptr::element_type, BinaryOperation>::value;
  detail::for_each(g, first, last, [&](const typename Ptr::element_type &x) {
    partial = binary_op(partial, x);
  });
  return reduce(g, partial, binary_op);
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename Ptr, typename T, class BinaryOperation>
EnableIfIsPointer<Ptr, T> reduce(Group g, Ptr first, Ptr last, T init,
                                 BinaryOperation binary_op) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(init, *first)), T>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(init, *first)), float>::value),
      "Result type of binary_op must match reduction accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  T partial =
      detail::identity<typename Ptr::element_type, BinaryOperation>::value;
  detail::for_each(g, first, last, [&](const typename Ptr::element_type &x) {
    partial = binary_op(partial, x);
  });
  return reduce(g, partial, init, binary_op);
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, class BinaryOperation>
EnableIfIsScalarArithmetic<T> exclusive_scan(Group g, T x,
                                             BinaryOperation binary_op) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  // FIXME: Do not special-case for half precision
  static_assert(std::is_same<decltype(binary_op(x, x)), T>::value ||
                    (std::is_same<T, half>::value &&
                     std::is_same<decltype(binary_op(x, x)), float>::value),
                "Result type of binary_op must match scan accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  return detail::calc<T, __spv::GroupOperation::ExclusiveScan,
                      detail::spirv::group_scope<Group>::value>(
      typename detail::GroupOpTag<T>::type(), x, binary_op);
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename T, class BinaryOperation>
EnableIfIsVectorArithmetic<T> exclusive_scan(Group g, T x,
                                             BinaryOperation binary_op) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(x[0], x[0])),
                   typename T::element_type>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(x[0], x[0])), float>::value),
      "Result type of binary_op must match scan accumulation type.");
  T result;
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = exclusive_scan(g, x[s], binary_op);
  }
  return result;
}

template <typename Group, typename V, typename T, class BinaryOperation>
EnableIfIsVectorArithmetic<T> exclusive_scan(Group g, V x, T init,
                                             BinaryOperation binary_op) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(init[0], x[0])),
                   typename T::element_type>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(init[0], x[0])), float>::value),
      "Result type of binary_op must match scan accumulation type.");
  T result;
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = exclusive_scan(g, x[s], init[s], binary_op);
  }
  return result;
}

template <typename Group, typename V, typename T, class BinaryOperation>
EnableIfIsScalarArithmetic<T> exclusive_scan(Group g, V x, T init,
                                             BinaryOperation binary_op) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  // FIXME: Do not special-case for half precision
  static_assert(std::is_same<decltype(binary_op(init, x)), T>::value ||
                    (std::is_same<T, half>::value &&
                     std::is_same<decltype(binary_op(init, x)), float>::value),
                "Result type of binary_op must match scan accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  typename Group::linear_id_type local_linear_id =
      detail::get_local_linear_id(g);
  if (local_linear_id == 0) {
    x = binary_op(init, x);
  }
  T scan = exclusive_scan(g, x, binary_op);
  if (local_linear_id == 0) {
    scan = init;
  }
  return scan;
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename InPtr, typename OutPtr, typename T,
          class BinaryOperation>
EnableIfIsPointer<InPtr, OutPtr>
exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result, T init,
               BinaryOperation binary_op) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(*first, *first)), T>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(*first, *first)), float>::value),
      "Result type of binary_op must match scan accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  ptrdiff_t offset = detail::get_local_linear_id(g);
  ptrdiff_t stride = detail::get_local_linear_range(g);
  ptrdiff_t N = last - first;
  auto roundup = [=](const ptrdiff_t &v,
                     const ptrdiff_t &divisor) -> ptrdiff_t {
    return ((v + divisor - 1) / divisor) * divisor;
  };
  typename InPtr::element_type x;
  typename OutPtr::element_type carry = init;
  for (ptrdiff_t chunk = 0; chunk < roundup(N, stride); chunk += stride) {
    ptrdiff_t i = chunk + offset;
    if (i < N) {
      x = first[i];
    }
    typename OutPtr::element_type out = exclusive_scan(g, x, carry, binary_op);
    if (i < N) {
      result[i] = out;
    }
    carry = broadcast(g, binary_op(out, x), stride - 1);
  }
  return result + N;
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation>
EnableIfIsPointer<InPtr, OutPtr> exclusive_scan(Group g, InPtr first,
                                                InPtr last, OutPtr result,
                                                BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(*first, *first)),
                   typename OutPtr::element_type>::value ||
          (std::is_same<typename OutPtr::element_type, half>::value &&
           std::is_same<decltype(binary_op(*first, *first)), float>::value),
      "Result type of binary_op must match scan accumulation type.");
  return exclusive_scan(
      g, first, last, result,
      detail::identity<typename OutPtr::element_type, BinaryOperation>::value,
      binary_op);
}

template <typename Group, typename T, class BinaryOperation>
EnableIfIsVectorArithmetic<T> inclusive_scan(Group g, T x,
                                             BinaryOperation binary_op) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(x[0], x[0])),
                   typename T::element_type>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(x[0], x[0])), float>::value),
      "Result type of binary_op must match scan accumulation type.");
  T result;
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = inclusive_scan(g, x[s], binary_op);
  }
  return result;
}

template <typename Group, typename T, class BinaryOperation>
EnableIfIsScalarArithmetic<T> inclusive_scan(Group g, T x,
                                             BinaryOperation binary_op) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  // FIXME: Do not special-case for half precision
  static_assert(std::is_same<decltype(binary_op(x, x)), T>::value ||
                    (std::is_same<T, half>::value &&
                     std::is_same<decltype(binary_op(x, x)), float>::value),
                "Result type of binary_op must match scan accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  return detail::calc<T, __spv::GroupOperation::InclusiveScan,
                      detail::spirv::group_scope<Group>::value>(
      typename detail::GroupOpTag<T>::type(), x, binary_op);
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename V, class BinaryOperation, typename T>
EnableIfIsScalarArithmetic<T>
inclusive_scan(Group g, V x, BinaryOperation binary_op, T init) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  // FIXME: Do not special-case for half precision
  static_assert(std::is_same<decltype(binary_op(init, x)), T>::value ||
                    (std::is_same<T, half>::value &&
                     std::is_same<decltype(binary_op(init, x)), float>::value),
                "Result type of binary_op must match scan accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  if (detail::get_local_linear_id(g) == 0) {
    x = binary_op(init, x);
  }
  return inclusive_scan(g, x, binary_op);
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename V, class BinaryOperation, typename T>
EnableIfIsVectorArithmetic<T>
inclusive_scan(Group g, V x, BinaryOperation binary_op, T init) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(init[0], x[0])), T>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(init[0], x[0])), float>::value),
      "Result type of binary_op must match scan accumulation type.");
  T result;
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = inclusive_scan(g, x[s], binary_op, init[s]);
  }
  return result;
}

template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation, typename T>
EnableIfIsPointer<InPtr, OutPtr>
inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
               BinaryOperation binary_op, T init) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(init, *first)), T>::value ||
          (std::is_same<T, half>::value &&
           std::is_same<decltype(binary_op(init, *first)), float>::value),
      "Result type of binary_op must match scan accumulation type.");
#ifdef __SYCL_DEVICE_ONLY__
  ptrdiff_t offset = detail::get_local_linear_id(g);
  ptrdiff_t stride = detail::get_local_linear_range(g);
  ptrdiff_t N = last - first;
  auto roundup = [=](const ptrdiff_t &v,
                     const ptrdiff_t &divisor) -> ptrdiff_t {
    return ((v + divisor - 1) / divisor) * divisor;
  };
  typename InPtr::element_type x;
  typename OutPtr::element_type carry = init;
  for (ptrdiff_t chunk = 0; chunk < roundup(N, stride); chunk += stride) {
    ptrdiff_t i = chunk + offset;
    if (i < N) {
      x = first[i];
    }
    typename OutPtr::element_type out = inclusive_scan(g, x, binary_op, carry);
    if (i < N) {
      result[i] = out;
    }
    carry = broadcast(g, out, stride - 1);
  }
  return result + N;
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

template <typename Group, typename InPtr, typename OutPtr,
          class BinaryOperation>
EnableIfIsPointer<InPtr, OutPtr> inclusive_scan(Group g, InPtr first,
                                                InPtr last, OutPtr result,
                                                BinaryOperation binary_op) {
  // FIXME: Do not special-case for half precision
  static_assert(
      std::is_same<decltype(binary_op(*first, *first)),
                   typename OutPtr::element_type>::value ||
          (std::is_same<typename OutPtr::element_type, half>::value &&
           std::is_same<decltype(binary_op(*first, *first)), float>::value),
      "Result type of binary_op must match scan accumulation type.");
  return inclusive_scan(
      g, first, last, result, binary_op,
      detail::identity<typename OutPtr::element_type, BinaryOperation>::value);
}

template <typename Group> bool leader(Group g) {
  static_assert(detail::is_generic_group<Group>::value,
                "Group algorithms only support the sycl::group and "
                "intel::sub_group class.");
#ifdef __SYCL_DEVICE_ONLY__
  typename Group::linear_id_type linear_id = detail::get_local_linear_id(g);
  return (linear_id == 0);
#else
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
#endif
}

} // namespace intel
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
#endif // __DISABLE_SYCL_INTEL_GROUP_ALGORITHMS__
