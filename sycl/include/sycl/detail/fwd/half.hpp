//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>

namespace sycl {
inline namespace _V1 {
namespace detail::half_impl {
class half;

// Several aliases are defined below:
// - StorageT: actual representation of half data type. It is used by scalar
//   half values. On device side, it points to some native half data type, while
//   on host it is represented by a 16-bit integer that the implementation
//   manipulates to emulate half-precision floating-point behavior.
//
// - BIsRepresentationT: data type which is used by built-in functions. It is
//   distinguished from StorageT, because on host, we can still operate on the
//   wrapper itself and there is no sense in direct usage of underlying data
//   type (too many changes required for BIs implementation without any
//   foreseeable profits)
//
// - VecElemT: representation of each element in the vector. On device it is
//   the same as StorageT to carry a native vector representation, while on
//   host it stores the sycl::half implementation directly.
//
// - VecNStorageT: representation of N-element vector of halfs. Follows the
//   same logic as VecElemT.
#ifdef __SYCL_DEVICE_ONLY__
using StorageT = _Float16;
using BIsRepresentationT = _Float16;
using VecElemT = _Float16;
#else  // SYCL_DEVICE_ONLY
using StorageT = uint16_t;
// No need to extract underlying data type for built-in functions operating on
// host
using BIsRepresentationT = half;
using VecElemT = half;
#endif // SYCL_DEVICE_ONLY
} // namespace detail::half_impl
using half = detail::half_impl::half;

} // namespace _V1
} // namespace sycl
