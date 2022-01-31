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

/// @defgroup sycl_esimd DPC++ Explicit SIMD API
/// This is a low-level API providing direct access to Intel GPU hardware
/// features. ESIMD overview can be found
/// [here](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/SYCL_EXT_INTEL_ESIMD/SYCL_EXT_INTEL_ESIMD.md).

///@{
/// @ingroup sycl_esimd

/// @defgroup sycl_esimd_core ESIMD core.
/// Core APIs defining main vector data types and their interfaces.

/// @defgroup sycl_esimd_memory Memory access API.
/// ESIMD APIs to access memory via accessors, USM pointers, perform per-element
/// atomic operations.

/// @defgroup sycl_esimd_math ESIMD math operations.
/// Defines math operations on ESIMD vector data types.

/// @defgroup sycl_esimd_bitmanip Bit and mask manipulation APIs.

/// @defgroup sycl_esimd_conv Explicit conversions.
/// @ingroup sycl_esimd
/// Defines explicit conversions (with and without saturation), truncation etc.
/// between ESIMD vector types.

/// @defgroup sycl_esimd_misc Miscellaneous ESIMD convenience functions.

/// The main components of the API are:
///   - @ref sycl_esimd_core - core API defining main vector data types and
///   their
///     interfaces.
///   - @ref sycl_esimd_memory
///   - @ref sycl_esimd_math
///   - @ref sycl_esimd_bitmanip
///   - @ref sycl_esimd_conv
///   - @ref sycl_esimd_misc
///@}

#include <sycl/ext/intel/experimental/esimd/alt_ui.hpp>
#include <sycl/ext/intel/experimental/esimd/common.hpp>
#include <sycl/ext/intel/experimental/esimd/math.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/intel/experimental/esimd/simd.hpp>
#include <sycl/ext/intel/experimental/esimd/simd_view.hpp>
