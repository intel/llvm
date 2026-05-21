//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <type_traits> // for enable_if_t, remove_const_t

namespace sycl {
inline namespace _V1 {
namespace detail {
template <typename DataT> class aligned_allocator;
} // namespace detail

// The non-type template parameter is intentionally named "Dimensions" here
// even though the primary template in <sycl/buffer.hpp> spells it
// "dimensions". Param names need not match across declarations, and MSVC
// /permissive- (Visual Studio 2022 17.x) miscompiles the lowercase form
// when the default for `__Enabled` references it from this forward
// declaration ("error C2065: 'dimensions': undeclared identifier" while
// instantiating sycl::accessor). The capitalized name avoids the MSVC
// lookup bug; behavior on GCC and Clang is unchanged.
template <typename T, int Dimensions = 1,
          typename AllocatorT =
              detail::aligned_allocator<std::remove_const_t<T>>,
          typename __Enabled =
              std::enable_if_t<(Dimensions > 0) && (Dimensions <= 3)>>
class buffer;
} // namespace _V1
} // namespace sycl
