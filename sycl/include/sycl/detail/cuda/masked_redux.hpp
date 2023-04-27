//==----- masked_redux.hpp - cuda masked reduction builtins and impls  -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/type_traits.hpp>
#include <sycl/group.hpp>
#include <sycl/detail/spirv.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
  namespace detail {

template <typename T, class BinaryOperation>
using IsRedux =
    std::bool_constant<std::is_integral<T>::value &&
                           sycl::detail::IsBitAND<T, BinaryOperation>::value ||
                       sycl::detail::IsBitOR<T, BinaryOperation>::value ||
                       sycl::detail::IsBitXOR<T, BinaryOperation>::value ||
                       sycl::detail::IsPlus<T, BinaryOperation>::value ||
                       sycl::detail::IsMinimum<T, BinaryOperation>::value ||
                       sycl::detail::IsMaximum<T, BinaryOperation>::value>;

#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__)

//// Masked reductions using redux.sync, requires integer types

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<sycl::detail::is_sugeninteger<T>::value &&
                     sycl::detail::IsMinimum<T, BinaryOperation>::value,
                 T>
masked_reduction_cuda_sm80(Group g, T x, BinaryOperation binary_op,
                           const uint32_t MemberMask) {
  return __nvvm_redux_sync_umin(x, MemberMask);
}

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<sycl::detail::is_sigeninteger<T>::value &&
                     sycl::detail::IsMinimum<T, BinaryOperation>::value,
                 T>
masked_reduction_cuda_sm80(Group g, T x, BinaryOperation binary_op,
                           const uint32_t MemberMask) {
  return __nvvm_redux_sync_min(x, MemberMask);
}

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<sycl::detail::is_sugeninteger<T>::value &&
                     sycl::detail::IsMaximum<T, BinaryOperation>::value,
                 T>
masked_reduction_cuda_sm80(Group g, T x, BinaryOperation binary_op,
                           const uint32_t MemberMask) {
  return __nvvm_redux_sync_umax(x, MemberMask);
}

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<sycl::detail::is_sigeninteger<T>::value &&
                     sycl::detail::IsMaximum<T, BinaryOperation>::value,
                 T>
masked_reduction_cuda_sm80(Group g, T x, BinaryOperation binary_op,
                           const uint32_t MemberMask) {
  return __nvvm_redux_sync_max(x, MemberMask);
}

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<(sycl::detail::is_sugeninteger<T>::value ||
                  sycl::detail::is_sigeninteger<T>::value) &&
                     sycl::detail::IsPlus<T, BinaryOperation>::value,
                 T>
masked_reduction_cuda_sm80(Group g, T x, BinaryOperation binary_op,
                           const uint32_t MemberMask) {
  return __nvvm_redux_sync_add(x, MemberMask);
}

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<(sycl::detail::is_sugeninteger<T>::value ||
                  sycl::detail::is_sigeninteger<T>::value) &&
                     sycl::detail::IsBitAND<T, BinaryOperation>::value,
                 T>
masked_reduction_cuda_sm80(Group g, T x, BinaryOperation binary_op,
                           const uint32_t MemberMask) {
  return __nvvm_redux_sync_and(x, MemberMask);
}

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<(sycl::detail::is_sugeninteger<T>::value ||
                  sycl::detail::is_sigeninteger<T>::value) &&
                     sycl::detail::IsBitOR<T, BinaryOperation>::value,
                 T>
masked_reduction_cuda_sm80(Group g, T x, BinaryOperation binary_op,
                           const uint32_t MemberMask) {
  return __nvvm_redux_sync_or(x, MemberMask);
}

template <typename Group, typename T, class BinaryOperation>
std::enable_if_t<(sycl::detail::is_sugeninteger<T>::value ||
                  sycl::detail::is_sigeninteger<T>::value) &&
                     sycl::detail::IsBitXOR<T, BinaryOperation>::value,
                 T>
masked_reduction_cuda_sm80(Group g, T x, BinaryOperation binary_op,
                           const uint32_t MemberMask) {
  return __nvvm_redux_sync_xor(x, MemberMask);
}
////

//// Shuffle based masked reduction impls

// Cluster group reduction using shfls, T = double
template <typename Group, typename T, class BinaryOperation>
inline __SYCL_ALWAYS_INLINE std::enable_if_t<
    ext::oneapi::experimental::is_cluster_group<Group>::value &&
        std::is_same_v<T, double>,
    T>
masked_reduction_cuda_shfls(Group g, T x, BinaryOperation binary_op,
                            const uint32_t MemberMask) {
  for (int i = g.get_local_range()[0] / 2; i > 0; i /= 2) {
    int x_a, x_b;
    asm volatile("mov.b64 {%0,%1},%2; \n\t" : "=r"(x_a), "=r"(x_b) : "l"(x));

    auto tmp_a = __nvvm_shfl_sync_bfly_i32(MemberMask, x_a, -1, i);
    auto tmp_b = __nvvm_shfl_sync_bfly_i32(MemberMask, x_b, -1, i);
    double tmp;
    asm volatile("mov.b64 %0,{%1,%2}; \n\t"
                 : "=l"(tmp)
                 : "r"(tmp_a), "r"(tmp_b));
    x = binary_op(x, tmp);
  }

  return x;
}

// Cluster group reduction using shfls, T = float
template <typename Group, typename T, class BinaryOperation>
inline __SYCL_ALWAYS_INLINE std::enable_if_t<
    ext::oneapi::experimental::is_cluster_group<Group>::value &&
        std::is_same_v<T, float>,
    T>
masked_reduction_cuda_shfls(Group g, T x, BinaryOperation binary_op,
                            const uint32_t MemberMask) {

  for (int i = g.get_local_range()[0] / 2; i > 0; i /= 2) {
    auto tmp =
        __nvvm_shfl_sync_bfly_i32(MemberMask, __nvvm_bitcast_f2i(x), -1, i);
    x = binary_op(x, __nvvm_bitcast_i2f(tmp));
  }
  return x;
}

// Cluster group reduction using shfls, std::is_integral_v<T>
template <typename Group, typename T, class BinaryOperation>
inline __SYCL_ALWAYS_INLINE std::enable_if_t<
    ext::oneapi::experimental::is_cluster_group<Group>::value &&
        std::is_integral_v<T>,
    T>
masked_reduction_cuda_shfls(Group g, T x, BinaryOperation binary_op,
                            const uint32_t MemberMask) {//todo membermask naming?

  for (int i = g.get_local_range()[0] / 2; i > 0; i /= 2) {
    auto tmp = __nvvm_shfl_sync_bfly_i32(MemberMask, x, -1, i);
    x = binary_op(x, tmp);
  }
  return x;
}

// Opportunistic/Ballot group reduction using shfls
// TODO in some places it might make sense to factor out parts of this big
// function.
template <typename Group, typename T, class BinaryOperation>
inline __SYCL_ALWAYS_INLINE std::enable_if_t<
    ext::oneapi::experimental::is_user_constructed_group_v<Group> &&
        !ext::oneapi::experimental::is_cluster_group<Group>::value,
    T>
masked_reduction_cuda_shfls(Group g, T x, BinaryOperation binary_op,
                            const uint32_t MemberMask) {

  if (MemberMask == 0xffffffff) {
    for (int i = 16; i > 0; i /= 2) {

      auto tmp = __nvvm_shfl_sync_bfly_i32(MemberMask, x, -1, i);
      x = binary_op(x, tmp);
    }
    return x;
  }
  unsigned local_range;
  // get_local_range()[0] in a more direct way.
  asm("popc.b32 %0, %1;" : "=r"(local_range) : "r"(MemberMask));

  // position of this lanes set bit with respect to all set bits in mask
  // local_set_bit = 1 for first set bit in mask.
  unsigned local_set_bit;

  // get_local_id()[0] directly without duplicating extract mask.
  asm("popc.b32 %0, %1;"
      : "=r"(local_set_bit)
      : "r"(MemberMask & __nvvm_read_ptx_sreg_lanemask_lt()));
  local_set_bit++;
  if (local_range < 2) {
    return x;
  }

  // number of elements remaining requiring binary operations
  auto op_range = local_range;

  // remainder that won't have a binary partner each pass of while loop
  int remainder;

  while (op_range / 2 >= 1) {
    remainder = op_range % 2;

    // stride between local_ids forming a binary op
    int stride = op_range / 2;

    // position of set bit in mask from shfl src lane.
    int src_set_bit;

    int unfold = local_set_bit + stride;
    bool fold_around = unfold > local_range;

    if (remainder != 0) {
      if (fold_around) {
        unfold++;
        src_set_bit = unfold - local_range;
      } else if (local_set_bit == 1) {
        src_set_bit = local_set_bit;
      } else {
        src_set_bit = unfold;
      }
    } else if (fold_around) {
      src_set_bit = unfold - local_range;
    } else {
      src_set_bit = unfold;
    }

    T tmp;
    // TODO adsorb these guys into separate functions since we call each form
    // twice.
    if (std::is_same_v<T, double>) {
      int x_a, x_b;
      asm volatile("mov.b64 {%0,%1},%2; \n\t" : "=r"(x_a), "=r"(x_b) : "l"(x));

      auto tmp_a = __nvvm_shfl_sync_idx_i32(
          MemberMask, x_a, __nvvm_fns(MemberMask, 0, src_set_bit), 0x1f);
      auto tmp_b = __nvvm_shfl_sync_idx_i32(
          MemberMask, x_b, __nvvm_fns(MemberMask, 0, src_set_bit), 0x1f);
      asm volatile("mov.b64 %0,{%1,%2}; \n\t"
                   : "=l"(tmp)
                   : "r"(tmp_a), "r"(tmp_b));
    } else {
      auto input = std::is_same_v<T, float> ? __nvvm_bitcast_f2i(x) : x;
      auto tmp_b32 = __nvvm_shfl_sync_idx_i32(
          MemberMask, input, __nvvm_fns(MemberMask, 0, src_set_bit), 0x1f);
      tmp = std::is_same_v<T, float> ? __nvvm_bitcast_i2f(tmp_b32) : tmp_b32;
    }
    x = (local_set_bit == 1 && remainder != 0) ? x : binary_op(x, tmp);

    op_range = std::ceil((float)op_range / 2.0f);
  }

  int broadID;
  int maskRev;
  asm("brev.b32 %0, %1;" : "=r"(maskRev) : "r"(MemberMask));
  asm("clz.b32 %0, %1;" : "=r"(broadID) : "r"(maskRev));

  T res;

  if (std::is_same_v<T, double>) {

    int x_a, x_b;
    asm volatile("mov.b64 {%0,%1},%2; \n\t" : "=r"(x_a), "=r"(x_b) : "l"(x));

    auto tmp_a = __nvvm_shfl_sync_idx_i32(MemberMask, x_a, broadID, 0x1f);
    auto tmp_b = __nvvm_shfl_sync_idx_i32(MemberMask, x_b, broadID, 0x1f);
    asm volatile("mov.b64 %0,{%1,%2}; \n\t"
                 : "=l"(res)
                 : "r"(tmp_a), "r"(tmp_b));

  } else {
    auto input = std::is_same_v<T, float> ? __nvvm_bitcast_f2i(x) : x;
    auto tmp_b32 = __nvvm_shfl_sync_idx_i32(MemberMask, input, broadID, 0x1f);
    res = std::is_same_v<T, float> ? __nvvm_bitcast_i2f(tmp_b32) : tmp_b32;
  }

  return res;
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
////

#endif
#endif
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
