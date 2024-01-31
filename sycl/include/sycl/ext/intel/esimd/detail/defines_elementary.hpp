//==---------------- defines_elementary.hpp - DPC++ Explicit SIMD API ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Elementary definitions used in Explicit SIMD APIs.
//===----------------------------------------------------------------------===//

#pragma once

/// @cond ESIMD_DETAIL

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_ESIMD_KERNEL __attribute__((sycl_explicit_simd))
#define SYCL_ESIMD_FUNCTION __attribute__((sycl_explicit_simd))

// Mark a function being nodebug.
#define ESIMD_NODEBUG __attribute__((nodebug))
// Mark a "ESIMD global": accessible from all functions in current translation
// unit, separate copy per subgroup (work-item), mapped to SPIR-V private
// storage class.
#define ESIMD_PRIVATE                                                          \
  __attribute__((opencl_private)) __attribute__((sycl_explicit_simd))
// Bind a ESIMD global variable to a specific register.
#define ESIMD_REGISTER(n) __attribute__((register_num(n)))

#define __ESIMD_API ESIMD_NODEBUG ESIMD_INLINE
#else // __SYCL_DEVICE_ONLY__
#define SYCL_ESIMD_KERNEL
#define SYCL_ESIMD_FUNCTION

// TODO ESIMD define what this means on Windows host
#define ESIMD_NODEBUG
// On host device ESIMD global is a thread local static var. This assumes that
// each work-item is mapped to a separate OS thread on host device.
#define ESIMD_PRIVATE thread_local
#define ESIMD_REGISTER(n)
#ifdef __ESIMD_BUILD_HOST_CODE
#define __ESIMD_API ESIMD_INLINE
#else // __ESIMD_BUILD_HOST_CODE
#define __ESIMD_API ESIMD_NOINLINE __attribute__((internal_linkage))
#endif // __ESIMD_BUILD_HOST_CODE
#endif // __SYCL_DEVICE_ONLY__

// Mark a function being noinline
#define ESIMD_NOINLINE __attribute__((noinline))
// Force a function to be inlined. 'inline' is used to preserve ODR for
// functions defined in a header.
#define ESIMD_INLINE inline __attribute__((always_inline))

// Macros for internal use
#define __ESIMD_NS sycl::ext::intel::esimd
#define __ESIMD_DNS sycl::ext::intel::esimd::detail
#define __ESIMD_ENS sycl::ext::intel::experimental::esimd
#define __ESIMD_EDNS sycl::ext::intel::experimental::esimd::detail
#define __ESIMD_XMX_NS sycl::ext::intel::esimd::xmx
#define __ESIMD_XMX_DNS sycl::ext::intel::esimd::xmx::detail

#define __ESIMD_QUOTE1(m) #m
#define __ESIMD_QUOTE(m) __ESIMD_QUOTE1(m)
#define __ESIMD_NS_QUOTED __ESIMD_QUOTE(__ESIMD_NS)
#define __ESIMD_DEPRECATED(new_api)                                            \
  __SYCL_DEPRECATED("use " __ESIMD_NS_QUOTED "::" __ESIMD_QUOTE(new_api))

/// @endcond ESIMD_DETAIL
