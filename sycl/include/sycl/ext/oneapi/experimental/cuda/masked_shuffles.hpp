//==--------- masked_shuffles.hpp - cuda masked shuffle algorithms ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)

namespace sycl {
inline namespace _V1 {
namespace detail {

#define CUDA_SHFL_SYNC(SHUFFLE_INSTR)                                          \
  template <typename T>                                                        \
  inline __SYCL_ALWAYS_INLINE T cuda_shfl_sync_##SHUFFLE_INSTR(                \
      unsigned int mask, T val, unsigned int shfl_param, int c) {              \
    T res;                                                                     \
    if constexpr (std::is_same_v<T, double>) {                                 \
      int x_a, x_b;                                                            \
      asm("mov.b64 {%0,%1},%2;" : "=r"(x_a), "=r"(x_b) : "d"(val));            \
      auto tmp_a = __nvvm_shfl_sync_##SHUFFLE_INSTR(mask, x_a, shfl_param, c); \
      auto tmp_b = __nvvm_shfl_sync_##SHUFFLE_INSTR(mask, x_b, shfl_param, c); \
      asm("mov.b64 %0,{%1,%2};" : "=d"(res) : "r"(tmp_a), "r"(tmp_b));         \
    } else if constexpr (std::is_same_v<T, long> ||                            \
                         std::is_same_v<T, unsigned long>) {                   \
      int x_a, x_b;                                                            \
      asm("mov.b64 {%0,%1},%2;" : "=r"(x_a), "=r"(x_b) : "l"(val));            \
      auto tmp_a = __nvvm_shfl_sync_##SHUFFLE_INSTR(mask, x_a, shfl_param, c); \
      auto tmp_b = __nvvm_shfl_sync_##SHUFFLE_INSTR(mask, x_b, shfl_param, c); \
      asm("mov.b64 %0,{%1,%2};" : "=l"(res) : "r"(tmp_a), "r"(tmp_b));         \
    } else if constexpr (std::is_same_v<T, half>) {                            \
      short tmp_b16;                                                           \
      asm("mov.b16 %0,%1;" : "=h"(tmp_b16) : "h"(val));                        \
      auto tmp_b32 = __nvvm_shfl_sync_##SHUFFLE_INSTR(                         \
          mask, static_cast<int>(tmp_b16), shfl_param, c);                     \
      asm("mov.b16 %0,%1;" : "=h"(res) : "h"(static_cast<short>(tmp_b32)));    \
    } else if constexpr (std::is_same_v<T, float>) {                           \
      auto tmp_b32 = __nvvm_shfl_sync_##SHUFFLE_INSTR(                         \
          mask, __nvvm_bitcast_f2i(val), shfl_param, c);                       \
      res = __nvvm_bitcast_i2f(tmp_b32);                                       \
    } else {                                                                   \
      res = __nvvm_shfl_sync_##SHUFFLE_INSTR(mask, val, shfl_param, c);        \
    }                                                                          \
    return res;                                                                \
  }

CUDA_SHFL_SYNC(bfly_i32)
CUDA_SHFL_SYNC(up_i32)
CUDA_SHFL_SYNC(down_i32)
CUDA_SHFL_SYNC(idx_i32)

#undef CUDA_SHFL_SYNC

} // namespace detail
} // namespace _V1
} // namespace sycl

#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
