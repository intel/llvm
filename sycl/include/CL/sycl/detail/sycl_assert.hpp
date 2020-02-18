//==----------- sycl_assert.hpp - SYCL standard header file ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/defines.hpp>

#include <cstdint>
#include <iostream>
#include <string>

#ifdef __SYCL_ENABLE_ASSERTIONS__
#ifndef __SYCL_DEVICE_ONLY__
#define __SYCL_STR(X) #X
#define __SYCL_XSTR(X) __SYCL_STR(X)
#define __SYCL_FILE__ __FILE__ ":" __SYCL_XSTR(__LINE__)
#define __SYCL_ASSERT(X, ...)                                                  \
  sycl::detail::sycl_assert(__SYCL_FILE__, X, __SYCL_XSTR(X), __VA_ARGS__)

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
template <typename... Args>
void sycl_assert(const std::string &FileName, bool Cond,
                 const std::string &CondStr, Args &&... Messages) {
  if (!Cond) {
    std::cerr << "Assertion \"" << CondStr << "\" at (" << FileName << "): ";
    using Expander = int[];
    (void)Expander{
        0, (void(std::cerr << std::forward<Args>(Messages) << " "), 0)...};
    exit(-1);
  }
}
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
#else // #ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_ASSERT(X, ...) assert(X)
#endif // #ifdef __SYCL_DEVICE_ONLY__
#else  // #ifdef __SYCL_ENABLE_ASSERTIONS__
#define __SYCL_ASSERT(X, ...)
#endif // #ifdef __SYCL_ENABLE_ASSERTIONS__

