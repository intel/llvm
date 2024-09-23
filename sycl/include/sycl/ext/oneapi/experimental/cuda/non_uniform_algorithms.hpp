//==----- non_uniform_algorithms.hpp - cuda masked subgroup algorithms -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
#include "masked_shuffles.hpp"

namespace sycl {
inline namespace _V1 {
namespace detail {

template <typename T, class BinaryOperation>
using IsRedux = std::bool_constant<
    std::is_integral<T>::value && IsBitAND<T, BinaryOperation>::value ||
    IsBitOR<T, BinaryOperation>::value || IsBitXOR<T, BinaryOperation>::value ||
    IsPlus<T, BinaryOperation>::value || IsMinimum<T, BinaryOperation>::value ||
    IsMaximum<T, BinaryOperation>::value>;

//// Masked reductions using redux.sync, requires integer types

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<is_sugeninteger_v<T> && IsMinimum<T, BinaryOperation>::value,
                 T>
masked_reduction_cuda_sm80(Group g, T x, BinaryOperation binary_op,
                           const uint32_t MemberMask) {
  return __nvvm_redux_sync_umin(x, MemberMask);
}

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<is_sigeninteger_v<T> && IsMinimum<T, BinaryOperation>::value,
                 T>
masked_reduction_cuda_sm80(Group g, T x, BinaryOperation binary_op,
                           const uint32_t MemberMask) {
  return __nvvm_redux_sync_min(x, MemberMask);
}

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<is_sugeninteger_v<T> && IsMaximum<T, BinaryOperation>::value,
                 T>
masked_reduction_cuda_sm80(Group g, T x, BinaryOperation binary_op,
                           const uint32_t MemberMask) {
  return __nvvm_redux_sync_umax(x, MemberMask);
}

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<is_sigeninteger_v<T> && IsMaximum<T, BinaryOperation>::value,
                 T>
masked_reduction_cuda_sm80(Group g, T x, BinaryOperation binary_op,
                           const uint32_t MemberMask) {
  return __nvvm_redux_sync_max(x, MemberMask);
}

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<(is_sugeninteger_v<T> ||
                  is_sigeninteger_v<T>)&&IsPlus<T, BinaryOperation>::value,
                 T>
masked_reduction_cuda_sm80(Group g, T x, BinaryOperation binary_op,
                           const uint32_t MemberMask) {
  return __nvvm_redux_sync_add(x, MemberMask);
}

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<(is_sugeninteger_v<T> ||
                  is_sigeninteger_v<T>)&&IsBitAND<T, BinaryOperation>::value,
                 T>
masked_reduction_cuda_sm80(Group g, T x, BinaryOperation binary_op,
                           const uint32_t MemberMask) {
  return __nvvm_redux_sync_and(x, MemberMask);
}

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<(is_sugeninteger_v<T> ||
                  is_sigeninteger_v<T>)&&IsBitOR<T, BinaryOperation>::value,
                 T>
masked_reduction_cuda_sm80(Group g, T x, BinaryOperation binary_op,
                           const uint32_t MemberMask) {
  return __nvvm_redux_sync_or(x, MemberMask);
}

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<(is_sugeninteger_v<T> ||
                  is_sigeninteger_v<T>)&&IsBitXOR<T, BinaryOperation>::value,
                 T>
masked_reduction_cuda_sm80(Group g, T x, BinaryOperation binary_op,
                           const uint32_t MemberMask) {
  return __nvvm_redux_sync_xor(x, MemberMask);
}
////

//// Shuffle based masked reduction impls

// fixed_size_group group reduction using shfls
template <typename Group, typename T, class BinaryOperation>
inline __SYCL_ALWAYS_INLINE std::enable_if_t<is_fixed_size_group_v<Group>, T>
masked_reduction_cuda_shfls(Group g, T x, BinaryOperation binary_op,
                            const uint32_t MemberMask) {
  for (int i = g.get_local_range()[0] / 2; i > 0; i /= 2) {
    T tmp = cuda_shfl_sync_bfly_i32(MemberMask, x, i, 0x1f);
    x = binary_op(x, tmp);
  }
  return x;
}

// Opportunistic/Ballot group reduction using shfls
template <typename Group, typename T, class BinaryOperation>
inline __SYCL_ALWAYS_INLINE std::enable_if_t<
    ext::oneapi::experimental::is_user_constructed_group_v<Group> &&
        !is_fixed_size_group_v<Group>,
    T>
masked_reduction_cuda_shfls(Group g, T x, BinaryOperation binary_op,
                            const uint32_t MemberMask) {

  unsigned localSetBit = g.get_local_id()[0] + 1;

  // number of elements requiring binary operations each loop iteration
  auto opRange = g.get_local_range()[0];

  // stride between local_ids forming a binary op
  unsigned stride = opRange / 2;
  while (stride >= 1) {

    // if (remainder == 1), there is a WI without a binary op partner
    unsigned remainder = opRange % 2;

    // unfolded position of set bit in mask of shfl src lane
    int unfoldedSrcSetBit = localSetBit + stride;

    // __nvvm_fns automatically wraps around to the correct bit position.
    // There is no performance impact on src_set_bit position wrt localSetBit
    T tmp = cuda_shfl_sync_idx_i32(
        MemberMask, x, __nvvm_fns(MemberMask, 0, unfoldedSrcSetBit), 31);

    if (!(localSetBit == 1 && remainder != 0)) {
      x = binary_op(x, tmp);
    }

    opRange = stride + remainder;
    stride = opRange / 2;
  }
  unsigned broadID;
  asm volatile(".reg .u32 rev;\n\t"
               "brev.b32 rev, %1;\n\t" // reverse mask bits
               "clz.b32 %0, rev;"
               : "=r"(broadID)
               : "r"(MemberMask));

  x = cuda_shfl_sync_idx_i32(MemberMask, x, broadID, 31);
  return x;
}

// Non Redux types must fall back to shfl based implementations.
template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<
    std::is_same<IsRedux<T, BinaryOperation>, std::false_type>::value &&
        ext::oneapi::experimental::is_user_constructed_group_v<Group>,
    T>
masked_reduction_cuda_sm80(Group g, T x, BinaryOperation binary_op,
                           const uint32_t MemberMask) {
  return masked_reduction_cuda_shfls(g, x, binary_op, MemberMask);
}

// get_identity is only currently used in this cuda specific header. If in the
// future it has more general use it should be moved to a more appropriate
// header.
template <typename T, class BinaryOperation>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<IsPlus<T, BinaryOperation>::value ||
                         IsBitOR<T, BinaryOperation>::value ||
                         IsBitXOR<T, BinaryOperation>::value,
                     T>
    get_identity() {
  return 0;
}

template <typename T, class BinaryOperation>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<IsMultiplies<T, BinaryOperation>::value, T>
    get_identity() {
  return 1;
}

template <typename T, class BinaryOperation>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<IsBitAND<T, BinaryOperation>::value, T>
    get_identity() {
  return ~0;
}

template <typename T, class BinaryOperation>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<IsMinimum<T, BinaryOperation>::value, T>
    get_identity() {
  return std::numeric_limits<T>::min();
}

template <typename T, class BinaryOperation>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<IsMaximum<T, BinaryOperation>::value, T>
    get_identity() {
  return std::numeric_limits<T>::max();
}

//// Shuffle based masked reduction impls

// fixed_size_group group scan using shfls
template <__spv::GroupOperation Op, typename Group, typename T,
          class BinaryOperation>
inline __SYCL_ALWAYS_INLINE std::enable_if_t<is_fixed_size_group_v<Group>, T>
masked_scan_cuda_shfls(Group g, T x, BinaryOperation binary_op,
                       const uint32_t MemberMask) {
  unsigned localIdVal = g.get_local_id()[0];
  for (int i = 1; i < g.get_local_range()[0]; i *= 2) {
    T tmp = cuda_shfl_sync_up_i32(MemberMask, x, i, 0);
    if (localIdVal >= i)
      x = binary_op(x, tmp);
  }
  if constexpr (Op == __spv::GroupOperation::ExclusiveScan) {
    x = cuda_shfl_sync_up_i32(MemberMask, x, 1, 0);
    if (localIdVal == 0) {
      return get_identity<T, BinaryOperation>();
    }
  }
  return x;
}

template <__spv::GroupOperation Op, typename Group, typename T,
          class BinaryOperation>
inline __SYCL_ALWAYS_INLINE std::enable_if_t<
    ext::oneapi::experimental::is_user_constructed_group_v<Group> &&
        !is_fixed_size_group_v<Group>,
    T>
masked_scan_cuda_shfls(Group g, T x, BinaryOperation binary_op,
                       const uint32_t MemberMask) {
  unsigned localIdVal = g.get_local_id()[0];
  unsigned localSetBit = localIdVal + 1;

  for (int i = 1; i < g.get_local_range()[0]; i *= 2) {
    int unfoldedSrcSetBit = localSetBit - i;

    T tmp = cuda_shfl_sync_idx_i32(
        MemberMask, x, __nvvm_fns(MemberMask, 0, unfoldedSrcSetBit), 31);

    if (localIdVal >= i)
      x = binary_op(x, tmp);
  }
  if constexpr (Op == __spv::GroupOperation::ExclusiveScan) {
    x = cuda_shfl_sync_idx_i32(MemberMask, x,
                               __nvvm_fns(MemberMask, 0, localSetBit - 1), 31);
    if (localIdVal == 0) {
      return get_identity<T, BinaryOperation>();
    }
  }
  return x;
}

} // namespace detail
} // namespace _V1
} // namespace sycl

#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
