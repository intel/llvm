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
#include <CL/sycl/group_algorithm.hpp>
#include <CL/sycl/nd_item.hpp>
#include <sycl/ext/oneapi/atomic.hpp>
#include <sycl/ext/oneapi/functional.hpp>
#include <sycl/ext/oneapi/sub_group.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {

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

#if (SYCL_EXT_ONEAPI_ASYNC_GROUP_COPY == 1)
namespace experimental {

template <typename Group,
          std::enable_if_t<sycl::is_group_v<Group>, bool> = true>
class async_copy_event {
public:
  using group = Group;
  __ocl_event_t Event;
  async_copy_event(__ocl_event_t Event) : Event(Event) {}
};

struct src_stride {
  std::size_t value;
};
struct dest_stride {
  std::size_t value;
};

namespace detail {
template <typename Group>
using is_sub_group = std::is_same<Group, sycl::ext::oneapi::sub_group>;

template <typename Group> constexpr auto group_to_scope() {
  if constexpr (is_sub_group<Group>::value) {
    return __spv::Scope::Subgroup;
  } else {
    return __spv::Scope::Workgroup;
  }
}

} // namespace detail

template <typename Group, typename... eventT>
std::enable_if_t<sycl::is_group_v<Group> &&
                 (std::is_same_v<eventT, async_copy_event<Group>> && ...)>
wait_for(Group, eventT... Events) {
  (__spirv_GroupWaitEvents(detail::group_to_scope<Group>(), 1, &Events.Event),
   ...);
}

/// Asynchronously copies a number of elements specified by \p numElements
/// from the source pointed by \p src to destination pointed by \p dest
/// with a source stride specified by \p srcStride, and returns an
/// async_copy_event which can be used to wait on the completion of the copy.
/// Permitted types for dataT are all scalar and vector types, except boolean.
template <typename Group, typename dataT>
std::enable_if_t<is_group_v<Group> && !sycl::detail::is_bool<dataT>::value,
                 async_copy_event<Group>>
joint_async_copy(Group g, global_ptr<dataT> src, local_ptr<dataT> dest,
                 size_t NumElements, src_stride SrcStride) {

  using DestT = sycl::detail::ConvertToOpenCLType_t<decltype(dest)>;
  using SrcT = sycl::detail::ConvertToOpenCLType_t<decltype(src)>;

  __ocl_event_t E = __SYCL_OpGroupAsyncCopyGlobalToLocal(
      detail::group_to_scope<Group>(), DestT(dest.get()), SrcT(src.get()),
      NumElements, SrcStride.value, 0);
  return async_copy_event<Group>(E);
}

/// Asynchronously copies a number of elements specified by \p numElements
/// from the source pointed by \p src to destination pointed by \p dest with
/// the destination stride specified by \p destStride, and returns an
/// async_copy_event which can be used to wait on the completion of the copy.
/// Permitted types for dataT are all scalar and vector types, except boolean.
template <typename Group, typename dataT>
std::enable_if_t<is_group_v<Group> && !sycl::detail::is_bool<dataT>::value,
                 async_copy_event<Group>>
joint_async_copy(Group g, local_ptr<dataT> src, global_ptr<dataT> dest,
                 size_t NumElements, dest_stride DestStride) {

  using DestT = sycl::detail::ConvertToOpenCLType_t<decltype(dest)>;
  using SrcT = sycl::detail::ConvertToOpenCLType_t<decltype(src)>;

  __ocl_event_t E = __SYCL_OpGroupAsyncCopyLocalToGlobal(
      detail::group_to_scope<Group>(), DestT(dest.get()), SrcT(src.get()),
      NumElements, DestStride.value, 0);
  return async_copy_event<Group>(E);
}

/// Specialization for bool type.
/// Asynchronously copies a number of elements specified by \p NumElements
/// from the source pointed by \p Src to destination pointed by \p Dest
/// with a stride specified by \p Stride, and returns an async_copy_event
/// which can be used to wait on the completion of the copy.
template <typename Group, typename dataT>
std::enable_if_t<is_group_v<Group> && sycl::detail::is_bool<dataT>::value,
                 async_copy_event<Group>>
joint_async_copy(Group g, global_ptr<dataT> Src, local_ptr<dataT> Dest,
                 size_t NumElements, src_stride srcStride) {
  static_assert(sizeof(bool) == sizeof(char),
                "Async copy to/from bool memory is not supported since sizeof "
                "bool is greater than 1 byte.");
  using BoolType =
      std::conditional_t<sycl::detail::is_vector_bool<dataT>::value,
                         sycl::detail::change_base_type_t<dataT, char>, char>;
  auto DestP = local_ptr<BoolType>(reinterpret_cast<BoolType *>(Dest.get()));
  auto SrcP = global_ptr<BoolType>(reinterpret_cast<BoolType *>(Src.get()));
  return joint_async_copy(g, SrcP, DestP, NumElements, srcStride);
}

/// Specialization for bool type.
/// Asynchronously copies a number of elements specified by \p NumElements
/// from the source pointed by \p Src to destination pointed by \p Dest
/// with a stride specified by \p Stride, and returns an async_copy_event
/// which can be used to wait on the completion of the copy.
template <typename Group, typename dataT>
std::enable_if_t<is_group_v<Group> && sycl::detail::is_bool<dataT>::value,
                 async_copy_event<Group>>
joint_async_copy(Group g, local_ptr<dataT> Src, global_ptr<dataT> Dest,
                 size_t NumElements, dest_stride destStride) {
  static_assert(sizeof(bool) == sizeof(char),
                "Async copy to/from bool memory is not supported since sizeof "
                "bool is greater than 1 byte.");
  using BoolType =
      std::conditional_t<sycl::detail::is_vector_bool<dataT>::value,
                         sycl::detail::change_base_type_t<dataT, char>, char>;
  auto DestP = global_ptr<BoolType>(reinterpret_cast<BoolType *>(Dest.get()));
  auto SrcP = local_ptr<BoolType>(reinterpret_cast<BoolType *>(Src.get()));
  return joint_async_copy(g, SrcP, DestP, NumElements, destStride);
}

/// Asynchronously copies a number of elements specified by \p numElements
/// from the source pointed by \p src to destination pointed by \p dest and
/// returns an async_copy_event which can be used to wait on the completion
/// of the copy.
/// Permitted types for dataT are all scalar and vector types.
template <typename Group, typename dataT>
std::enable_if_t<is_group_v<Group>, async_copy_event<Group>>
joint_async_copy(Group g, global_ptr<dataT> src, local_ptr<dataT> dest,
                 size_t numElements) {
  return joint_async_copy(g, src, dest, numElements, src_stride{1});
}

/// Asynchronously copies a number of elements specified by \p numElements
/// from the source pointed by \p src to destination pointed by \p dest and
/// returns an async_copy_event which can be used to wait on the completion
/// of the copy.
/// Permitted types for dataT are all scalar and vector types.
template <typename Group, typename dataT>
async_copy_event<Group> joint_async_copy(Group g, local_ptr<dataT> src,
                                         global_ptr<dataT> dest,
                                         size_t numElements) {
  return joint_async_copy(g, src, dest, numElements, dest_stride{1});
}

} // namespace experimental
#endif

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
                      PI_INVALID_DEVICE);
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
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = broadcast(g, x[s], local_id);
  }
  return result;
#else
  (void)g;
  (void)x;
  (void)local_id;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
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
                      PI_INVALID_DEVICE);
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
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = broadcast(g, x[s], linear_local_id);
  }
  return result;
#else
  (void)g;
  (void)x;
  (void)linear_local_id;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
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
                      PI_INVALID_DEVICE);
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
  for (int s = 0; s < x.get_size(); ++s) {
    result[s] = broadcast(g, x[s]);
  }
  return result;
#else
  (void)g;
  (void)x;
  throw runtime_error("Group algorithms are not supported on host device.",
                      PI_INVALID_DEVICE);
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

} // namespace oneapi
} // namespace ext

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
