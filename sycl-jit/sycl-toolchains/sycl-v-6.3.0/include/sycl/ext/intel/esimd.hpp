//==------------ esimd.hpp - DPC++ Explicit SIMD API -----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The main header of the Explicit SIMD API.
//===----------------------------------------------------------------------===//

#pragma once

// clang-format off
///
/// @defgroup sycl_esimd DPC++ Explicit SIMD API
/// This is a low-level API providing direct access to Intel GPU hardware
/// features. ESIMD overview can be found
/// [here](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md).
/// Some terminology used in the API documentation:
/// - *lane* -
///       (or "vector lane") Individual "lane" of input and output elements
///       in a ESIMD vector operation, such that all lanes combined for the
///       input and output vectors of the operation. Lane is indentified by
///       an ordinal in the [0, N-1) range, where N is the size of the
///       input/output vectors.
/// - *mask* -
///       a vector of predicates which can be used to enable/disable
///       execution of a vector operation over the correspondin lane.
///       \c 0 predicate value disables execution, non-zero - enables.
/// - *word* - 2 bytes.
/// - *dword* ("double word") - 4 bytes.
/// - *qword* ("quad word") - 8 bytes.
/// - *oword* ("octal word") - 16 bytes.
/// - *pixel* A 4 byte-aligned contiguous 128-bit chunk of memory logically
///    divided into 4 32-bit channels - \c R,\c G, \c B, \c A. Multiple pixels
///    can be accessed by ESIMD APIs, with ability to enable/disable access
///    to each channel for all pixels.
///
/// NOTES:
/// - API elements (macros, types, functions, etc.) starting with underscore
///   \c _, as well as those in \c detail namespace, are never supposed to be
///   used directly in the user code.

// clang-format on

/// @addtogroup sycl_esimd
/// @{

/// @defgroup sycl_esimd_core ESIMD core.
/// Core APIs defining main vector data types and their interfaces.

/// @defgroup sycl_esimd_memory Memory access API.
/// ESIMD APIs to access memory via accessors, USM pointers, perform per-element
/// atomic operations.

/// @defgroup sycl_esimd_math ESIMD math operations.
/// Defines math operations on ESIMD vector data types.

/// @defgroup sycl_esimd_bitmanip Bit and mask manipulation APIs.

/// @defgroup sycl_esimd_conv Explicit conversions.
/// Defines explicit conversions (with and without saturation), truncation etc.
/// between ESIMD vector types.

/// @defgroup sycl_esimd_raw_send Raw send APIs.
/// Implements the \c send instruction to send messages to variaous components
/// of the Intel(R) processor graphics, as defined in the documentation at
/// https://www.intel.com/content/www/us/en/docs/graphics-for-linux/developer-reference/1-0/hardware-specs.html

/// @defgroup sycl_esimd_misc Miscellaneous ESIMD convenience functions.

/// @} sycl_esimd

// TODO Disable "changes ABI" warnings emitted by host compiler, because ESIMD
// code is supposed to be compiled by clang++ only for now anyway, plus ESIMD
// code performance on host is not an issue yet.
#if !defined(__SYCL_DEVICE_ONLY__) && defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpsabi"
#endif // !defined(__SYCL_DEVICE_ONLY__) && defined(__clang__)

#include <sycl/detail/type_traits.hpp>
#include <sycl/ext/intel/esimd/alt_ui.hpp>
#include <sycl/ext/intel/esimd/common.hpp>
#include <sycl/ext/intel/esimd/detail/bfloat16_type_traits.hpp>
#include <sycl/ext/intel/esimd/detail/half_type_traits.hpp>
#include <sycl/ext/intel/esimd/detail/tfloat32_type_traits.hpp>
#include <sycl/ext/intel/esimd/memory_properties.hpp>
#include <sycl/ext/intel/esimd/simd.hpp>
#include <sycl/ext/intel/esimd/simd_view.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/math.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

#if !defined(__SYCL_DEVICE_ONLY__) && defined(__clang__)
#pragma clang diagnostics pop
#endif // !defined(__SYCL_DEVICE_ONLY__) && defined(__clang__)
