//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <type_traits> // for remove_const_t

namespace sycl {
inline namespace _V1 {
namespace detail {
template <typename DataT> class aligned_allocator;
} // namespace detail

// The trailing `__Enabled` template parameter is preserved (and defaults to
// `void`) to keep the mangled name of `buffer<T, Dims, AllocatorT>` identical
// to the historical 4-parameter form, which is observable in user code that
// declares functions taking buffers by value/reference. Removing the
// parameter would change ITANIUM/MSVC mangling for those user symbols.
//
// The previous SFINAE-based default for `__Enabled`
// (`std::enable_if_t<(Dimensions > 0) && (Dimensions <= 3)>`) was replaced
// with a non-dependent `void`. The dimension constraint is enforced by a
// `static_assert` inside the primary template in <sycl/buffer.hpp>, matching
// the convention used by other SYCL templates such as `range`, `id`, and
// `nd_range`. The dependent default expression also tripped a MSVC
// /permissive- bug when present at this forward declaration position
// (C2065 'Dimensions': undeclared identifier while instantiating
// sycl::accessor).
template <typename T, int Dimensions = 1,
          typename AllocatorT =
              detail::aligned_allocator<std::remove_const_t<T>>,
          typename __Enabled = void>
class buffer;
} // namespace _V1
} // namespace sycl
