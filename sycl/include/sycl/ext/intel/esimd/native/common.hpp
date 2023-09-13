//==-------------- native/memory.hpp - DPC++ Explicit SIMD API -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Explicit SIMD API types used in native ESIMD APIs.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>

#include <cstdint>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::esimd::native::lsc {

/// @addtogroup sycl_esimd_memory
/// @{

/// @defgroup sycl_esimd_memory_lsc LSC-specific memory access APIs.
/// This group combines types and functions specific to LSC, which is available
/// in Intel GPUs starting from PVC and ACM.

/// @} sycl_esimd_memory

/// @addtogroup sycl_esimd_memory_lsc
/// @{

// TODO move all LSC-related "common" APIs here

/// LSC atomic operation codes.
/// <tt>atomic_update<native::lsc::atomic_op::inc>(...);</tt> is a short-cut to
/// <tt>lsc_atomic_update<atomic_op::inc>(...);</tt> with default cache and data
/// size controls.
enum class atomic_op : uint8_t {
  inc = 0x08,      // atomic integer increment
  dec = 0x09,      // atomic integer decrement
  load = 0x0a,     // atomic load
  store = 0x0b,    // atomic store
  add = 0x0c,      // atomic integer add
  sub = 0x0d,      // atomic integer subtract
  smin = 0x0e,     // atomic signed int min
  smax = 0x0f,     // atomic signed int max
  umin = 0x10,     // atomic unsigned int min
  umax = 0x11,     // atomic unsigned int max
  cmpxchg = 0x12,  // atomic int compare and swap
  fadd = 0x13,     // floating-point add
  fsub = 0x14,     // floating-point subtract
  fmin = 0x15,     // floating-point min
  fmax = 0x16,     // floating-point max
  fcmpxchg = 0x17, // floating-point CAS
  bit_and = 0x18,  // logical (bitwise) AND
  bit_or = 0x19,   // logical (bitwise) OR
  bit_xor = 0x1a,  // logical (bitwise) XOR
};

/// @} sycl_esimd_memory_lsc

} // namespace ext::intel::esimd::native::lsc
} // namespace _V1
} // namespace sycl
