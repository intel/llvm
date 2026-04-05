//==---------------- vector.hpp --- Implements sycl::vec -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Check if Clang's ext_vector_type attribute is available. Host compiler
// may not be Clang, and Clang may not be built with the extension.
#ifdef __clang__
#ifndef __has_extension
#define __has_extension(x) 0
#endif
#ifndef __HAS_EXT_VECTOR_TYPE__
#if __has_extension(attribute_ext_vector_type)
#define __HAS_EXT_VECTOR_TYPE__
#endif
#endif
#endif // __clang__

// See vec::DataType definitions for more details
#ifndef __SYCL_USE_PLAIN_ARRAY_AS_VEC_STORAGE
#define __SYCL_USE_PLAIN_ARRAY_AS_VEC_STORAGE !__SYCL_USE_LIBSYCL8_VEC_IMPL
#endif

#if !defined(__HAS_EXT_VECTOR_TYPE__) && defined(__SYCL_DEVICE_ONLY__)
#error "SYCL device compiler is built without ext_vector_type support"
#endif

#include <sycl/detail/named_swizzles_mixin.hpp>
#include <sycl/detail/vector_base.hpp>
#include <sycl/detail/vector_arith.hpp>

#include <sycl/detail/common.hpp>
#include <sycl/detail/fwd/accessor.hpp>
#include <sycl/detail/fwd/half.hpp>
#include <sycl/detail/memcpy.hpp>

#include <sycl/detail/vector_swizzle.hpp>

#include <sycl/detail/vector_core.hpp>
#include <sycl/detail/vector_swizzle_op.hpp>

namespace sycl {
inline namespace _V1 {
} // namespace _V1
} // namespace sycl
