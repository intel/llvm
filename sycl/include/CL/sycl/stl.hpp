//==----------- stl.hpp - basic STL implementation -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// 4.5 C++ Standard library classes required for the interface

#include <CL/sycl/detail/defines.hpp>

#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

#if defined(_WIN32) && !defined(__SYCL_DEVICE_ONLY__)
namespace detail {
// SYCL library is designed such a way that STL objects cross DLL boundary,
// which is not guaranteed to work and considered not safe in general.
// Only using same dynamic C++ runtime library for sycl[d].dll and for
// the application using sycl[d].dll is guaranteed to work properly.
constexpr bool isMSVCDynamicCXXRuntime() {
// The options /MD and /MDd that make the code to use dynamic runtime also
// define the _DLL macro.
#ifdef _DLL
  return true;
#else
  return false;
#endif
}
static_assert(isMSVCDynamicCXXRuntime(),
              "SYCL library is designed to work with dynamic C++ runtime, "
              "please use /MD or /MDd switches.");
} // namespace detail
#endif // defined(_WIN32) && !defined(__SYCL_DEVICE_ONLY__)

template <class T, class Alloc = std::allocator<T>>
using vector_class = std::vector<T, Alloc>;

using string_class = std::string;

template <class Sig> using function_class = std::function<Sig>;

using mutex_class = std::mutex;

template <class T, class Deleter = std::default_delete<T>>
using unique_ptr_class = std::unique_ptr<T, Deleter>;

template <class T> using shared_ptr_class = std::shared_ptr<T>;

template <class T> using weak_ptr_class = std::weak_ptr<T>;

template <class T> using hash_class = std::hash<T>;

using exception_ptr_class = std::exception_ptr;

template <typename T, typename... ArgsT>
unique_ptr_class<T> make_unique_ptr(ArgsT &&... Args) {
  return unique_ptr_class<T>(new T(std::forward<ArgsT>(Args)...));
}

} // sycl
} // cl

