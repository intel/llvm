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

//// Generates all possible CUDA shfl.sync builtin calls.
#define CUDA_SHFL_SYNC(RES, MASK, VAL, SHFL_PARAM, C, SHUFFLE_INSTR)           \
if constexpr (std::is_same_v<T, double>) {                                     \
  int x_a, x_b;                                                                \
  asm("mov.b64 {%0,%1},%2;" : "=r"(x_a), "=r"(x_b) : "d"(VAL));                \
  auto tmp_a = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, x_a, SHFL_PARAM, C);     \
  auto tmp_b = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, x_b, SHFL_PARAM, C);     \
  asm("mov.b64 %0,{%1,%2};" : "=d"(RES) : "r"(tmp_a), "r"(tmp_b));             \
} else if constexpr (std::is_same_v<T, long> ||                                \
                     std::is_same_v<T, unsigned long>) {                       \
  int x_a, x_b;                                                                \
  asm("mov.b64 {%0,%1},%2;" : "=r"(x_a), "=r"(x_b) : "l"(VAL));                \
  auto tmp_a = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, x_a, SHFL_PARAM, C);     \
  auto tmp_b = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, x_b, SHFL_PARAM, C);     \
  asm("mov.b64 %0,{%1,%2};" : "=l"(RES) : "r"(tmp_a), "r"(tmp_b));             \
} else if constexpr (std::is_same_v<T, half>) {                                \
  short tmp_b16;                                                               \
  asm("mov.b16 %0,%1;" : "=h"(tmp_b16) : "h"(VAL));                            \
  auto tmp_b32 = __nvvm_shfl_sync_##SHUFFLE_INSTR(                             \
      MASK, static_cast<int>(tmp_b16), SHFL_PARAM, C);                         \
  asm("mov.b16 %0,%1;" : "=h"(RES) : "h"(static_cast<short>(tmp_b32)));        \
} else if constexpr (std::is_same_v<T, float>) {                               \
  auto tmp_b32 = __nvvm_shfl_sync_##SHUFFLE_INSTR(                             \
      MASK, __nvvm_bitcast_f2i(VAL), SHFL_PARAM, C);                           \
  RES = __nvvm_bitcast_i2f(tmp_b32);                                           \
} else {                                                                       \
  RES = __nvvm_shfl_sync_##SHUFFLE_INSTR(MASK, VAL, SHFL_PARAM, C);            \
}

} // namespace detail
} // namespace _V1
} // namespace sycl

#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
