//==- invoke_simd_types.hpp - SYCL invoke_simd extension types --*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
// Part of the implemenation of the sycl_ext_oneapi_invoke_simd extension.
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_invoke_simd.asciidoc
// ===--------------------------------------------------------------------=== //

#pragma once

// SYCL extension macro definition as required by the SYCL specification.
// 1 - Initial extension version. Base features are supported.
#define SYCL_EXT_ONEAPI_INVOKE_SIMD 1

#include <std/experimental/simd.hpp>
#include <sycl/detail/defines_elementary.hpp>

namespace sycl {
inline namespace _V1 {

namespace ext::oneapi::experimental {

// --- Basic definitions prescribed by the spec.
namespace simd_abi {
// "Fixed-size simd width of N" ABI based on clang vectors - used as the ABI for
// SIMD objects this implementation of invoke_simd spec is based on.
template <class T, int N>
using native_fixed_size = typename std::experimental::__simd_abi<
    std::experimental::_StorageKind::_VecExt, N>;
} // namespace simd_abi

// The SIMD object type, which is the generic std::experimental::simd type with
// the native fixed size ABI.
template <class T, int N>
using simd = std::experimental::simd<T, simd_abi::native_fixed_size<T, N>>;

// The SIMD mask object type.
template <class T, int N>
using simd_mask =
    std::experimental::simd_mask<T, simd_abi::native_fixed_size<T, N>>;
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
