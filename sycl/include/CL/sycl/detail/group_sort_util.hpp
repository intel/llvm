//==------------ group_sort_util.hpp ---------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file includes some utilities that are used by group sorting algorithms
//

#pragma once

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

#if __cplusplus < 201703
template <typename... _Ts> struct make_void_type { using type = void; };

template <typename... _Ts>
using void_type = typename make_void_type<_Ts...>::type;
#endif

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
