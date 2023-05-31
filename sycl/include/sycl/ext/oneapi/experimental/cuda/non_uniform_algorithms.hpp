//==----- non_uniform_algorithms.hpp - cuda masked subgroup algorithms -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <sycl/detail/spirv.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/group.hpp>

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

#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)

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
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<ext::oneapi::experimental::is_fixed_size_group_v<Group> &&
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
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<ext::oneapi::experimental::is_fixed_size_group_v<Group> &&
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
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<ext::oneapi::experimental::is_fixed_size_group_v<Group> &&
                         std::is_integral_v<T>,
                     T>
    masked_reduction_cuda_shfls(Group g, T x, BinaryOperation binary_op,
                                const uint32_t MemberMask) {

  for (int i = g.get_local_range()[0] / 2; i > 0; i /= 2) {
    auto tmp = __nvvm_shfl_sync_bfly_i32(MemberMask, x, -1, i);
    x = binary_op(x, tmp);
  }
  return x;
}

template <typename Group, typename T>
inline __SYCL_ALWAYS_INLINE std::enable_if_t<
    ext::oneapi::experimental::is_user_constructed_group_v<Group>, T>
non_uniform_shfl_T(const uint32_t MemberMask, T x, int delta) {
  if constexpr (ext::oneapi::experimental::is_fixed_size_group_v<Group>) {
    return __nvvm_shfl_sync_up_i32(MemberMask, x, delta, 0);
  } else {
    return __nvvm_shfl_sync_idx_i32(MemberMask, x, delta, 31);
  }
}

template <typename Group, typename T>
inline __SYCL_ALWAYS_INLINE std::enable_if_t<
    ext::oneapi::experimental::is_user_constructed_group_v<Group>, T>
non_uniform_shfl(Group g, const uint32_t MemberMask, T x, int delta) {
  T res;
  if constexpr (std::is_same_v<T, double>) {
    int x_a, x_b;
    asm volatile("mov.b64 {%0,%1},%2; \n\t" : "=r"(x_a), "=r"(x_b) : "l"(x));

    auto tmp_a = non_uniform_shfl_T<Group>(MemberMask, x_a, delta);
    auto tmp_b = non_uniform_shfl_T<Group>(MemberMask, x_b, delta);
    asm volatile("mov.b64 %0,{%1,%2}; \n\t"
                 : "=l"(res)
                 : "r"(tmp_a), "r"(tmp_b));
  } else {
    auto input = std::is_same_v<T, float> ? __nvvm_bitcast_f2i(x) : x;
    auto tmp_b32 = non_uniform_shfl_T<Group>(MemberMask, input, delta);
    res = std::is_same_v<T, float> ? __nvvm_bitcast_i2f(tmp_b32) : tmp_b32;
  }
  return res;
}

// Opportunistic/Ballot group reduction using shfls
template <typename Group, typename T, class BinaryOperation>
inline __SYCL_ALWAYS_INLINE std::enable_if_t<
    ext::oneapi::experimental::is_user_constructed_group_v<Group> &&
        !ext::oneapi::experimental::is_fixed_size_group_v<Group>,
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

  unsigned localSetBit = g.get_local_id()[0] + 1;

  // number of elements requiring binary operations each loop iteration
  auto opRange = g.get_local_range()[0];

  // remainder that won't have a binary partner each loop iteration
  int remainder;

  while (opRange / 2 >= 1) {
    remainder = opRange % 2;

    // stride between local_ids forming a binary op
    int stride = opRange / 2;

    // unfolded position of set bit in mask of shfl src lane
    int unfoldedSrcSetBit = localSetBit + stride;

    // __nvvm_fns automatically wraps around to the correct bit position.
    // There is no performance impact on src_set_bit position wrt localSetBit
    auto tmp = non_uniform_shfl(g, MemberMask, x,
                                __nvvm_fns(MemberMask, 0, unfoldedSrcSetBit));

    if (!(localSetBit == 1 && remainder != 0)) {
      x = binary_op(x, tmp);
    }

    opRange = std::ceil((float)opRange / 2.0f);
  }
  int broadID;
  int maskRev;
  asm volatile("brev.b32 %0, %1;" : "=r"(maskRev) : "r"(MemberMask));
  asm volatile("clz.b32 %0, %1;" : "=r"(broadID) : "r"(maskRev));

  return non_uniform_shfl(g, MemberMask, x, broadID);
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
    std::enable_if_t<sycl::detail::IsPlus<T, BinaryOperation>::value ||
                         sycl::detail::IsBitOR<T, BinaryOperation>::value ||
                         sycl::detail::IsBitXOR<T, BinaryOperation>::value,
                     T>
    get_identity() {
  return 0;
}

template <typename T, class BinaryOperation>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<sycl::detail::IsMultiplies<T, BinaryOperation>::value, T>
    get_identity() {
  return 1;
}

template <typename T, class BinaryOperation>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<sycl::detail::IsBitAND<T, BinaryOperation>::value, T>
    get_identity() {
  return ~0;
}

#define GET_ID(OP_CHECK, OP)                                                   \
  template <typename T, class BinaryOperation>                                 \
  inline __SYCL_ALWAYS_INLINE                                                  \
      std::enable_if_t<sycl::detail::OP_CHECK<T, BinaryOperation>::value, T>   \
      get_identity() {                                                         \
    if constexpr (std::is_same_v<T, char>) {                                   \
      return std::numeric_limits<char>::OP();                                  \
    } else if constexpr (std::is_same_v<T, unsigned char>) {                   \
      return std::numeric_limits<unsigned char>::OP();                         \
    } else if constexpr (std::is_same_v<T, short>) {                           \
      return std::numeric_limits<short>::OP();                                 \
    } else if constexpr (std::is_same_v<T, unsigned short>) {                  \
      return std::numeric_limits<unsigned short>::OP();                        \
    } else if constexpr (std::is_same_v<T, int>) {                             \
      return std::numeric_limits<int>::OP();                                   \
    } else if constexpr (std::is_same_v<T, unsigned int>) {                    \
      return std::numeric_limits<unsigned int>::OP();                          \
    } else if constexpr (std::is_same_v<T, long>) {                            \
      return std::numeric_limits<int>::OP();                                   \
    } else if constexpr (std::is_same_v<T, unsigned long>) {                   \
      return std::numeric_limits<unsigned int>::OP();                          \
    } else if constexpr (std::is_same_v<T, float>) {                           \
      return std::numeric_limits<float>::OP();                                 \
    } else if constexpr (std::is_same_v<T, double>) {                          \
      return std::numeric_limits<double>::OP();                                \
    }                                                                          \
    return 0;                                                                  \
  }

GET_ID(IsMinimum, max)
GET_ID(IsMaximum, min)

#undef GET_ID

//// Shuffle based masked reduction impls

// Cluster group scan using shfls
template <__spv::GroupOperation Op, typename Group, typename T,
          class BinaryOperation>
inline __SYCL_ALWAYS_INLINE
    std::enable_if_t<ext::oneapi::experimental::is_fixed_size_group_v<Group>, T>
    masked_scan_cuda_shfls(Group g, T x, BinaryOperation binary_op,
                           const uint32_t MemberMask) {
  for (int i = 1; i < g.get_local_range()[0]; i *= 2) {
    auto tmp = non_uniform_shfl(g, MemberMask, x, i);
    if (g.get_local_id()[0] >= i)
      x = binary_op(x, tmp);
  }
  if constexpr (Op == __spv::GroupOperation::ExclusiveScan) {

    x = non_uniform_shfl(g, MemberMask, x, 1);
    if (g.get_local_id()[0] == 0) {
      return get_identity<T, BinaryOperation>();
    }
  }
  return x;
}

template <__spv::GroupOperation Op, typename Group, typename T,
          class BinaryOperation>
inline __SYCL_ALWAYS_INLINE std::enable_if_t<
    ext::oneapi::experimental::is_user_constructed_group_v<Group> &&
        !ext::oneapi::experimental::is_fixed_size_group_v<Group>,
    T>
masked_scan_cuda_shfls(Group g, T x, BinaryOperation binary_op,
                       const uint32_t MemberMask) {
  int localIdVal = g.get_local_id()[0];
  int localSetBit = localIdVal + 1;

  for (int i = 1; i < g.get_local_range()[0]; i *= 2) {
    int unfoldedSrcSetBit = localSetBit - i;

    auto tmp = non_uniform_shfl(g, MemberMask, x,
                                __nvvm_fns(MemberMask, 0, unfoldedSrcSetBit));
    if (localIdVal >= i)
      x = binary_op(x, tmp);
  }
  if constexpr (Op == __spv::GroupOperation::ExclusiveScan) {
    x = non_uniform_shfl(g, MemberMask, x,
                         __nvvm_fns(MemberMask, 0, localSetBit - 1));
    if (localIdVal == 0) {
      return get_identity<T, BinaryOperation>();
    }
  }
  return x;
}

#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
