//==-------------- xmx/common.hpp - DPC++ Explicit SIMD API ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Explicit SIMD API types used in ESIMD Intel Xe Matrix eXtension.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::intel::esimd::xmx {

/// Describes the element types in the input matrices.
/// Used as template parameter to dpas() and may be omitted when
/// it is deducible from the element types of input matrices.
enum class dpas_argument_type {
  Invalid = 0,
  u1 = 1, // unsigned 1 bit
  U1 __SYCL_DEPRECATED("use u1") = u1,
  s1 = 2, // signed 1 bit
  S1 __SYCL_DEPRECATED("use s1") = s1,
  u2 = 3, // unsigned 2 bits
  U2 __SYCL_DEPRECATED("use u2") = u2,
  s2 = 4, // signed 2 bits
  S2 __SYCL_DEPRECATED("use s2") = s2,
  u4 = 5, // unsigned 4 bits
  U4 __SYCL_DEPRECATED("use u4") = u4,
  s4 = 6, // signed 4 bits
  S4 __SYCL_DEPRECATED("use s4") = s4,
  u8 = 7, // unsigned 8 bits
  U8 __SYCL_DEPRECATED("use u8") = u8,
  s8 = 8, // signed 8 bits
  S8 __SYCL_DEPRECATED("use s8") = s8,
  bf16 = 9, // bfloat 16
  BF16 __SYCL_DEPRECATED("use bf16") = bf16,
  fp16 = 10, // half float
  FP16 __SYCL_DEPRECATED("use fp16") = fp16,
  tf32 = 12, // tensorfloat 32
  TF32 __SYCL_DEPRECATED("use tf32") = tf32
};

} // namespace ext::intel::esimd::xmx
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
