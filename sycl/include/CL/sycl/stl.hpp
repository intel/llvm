//==----------- stl.hpp - basic STL implementation -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// 4.5 C++ Standard library classes required for the interface

#include <CL/sycl/bit_cast.hpp>
#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/sycl_span.hpp>

#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

#if defined(_WIN32) && !defined(_DLL) && !defined(__SYCL_DEVICE_ONLY__)
// SYCL library is designed such a way that STL objects cross DLL boundary,
// which is guaranteed to work properly only when the application uses the same
// C++ runtime that SYCL library uses.
// The appplications using sycl.dll must be linked with dynamic/release C++ MSVC
// runtime, i.e. be compiled with /MD switch. Similarly, the applications using
// sycld.dll must be linked with dynamic/debug C++ runtime and be compiled with
// /MDd switch.
// Compiler automatically adds /MD or /MDd when -fsycl switch is used.
// The options /MD and /MDd that make the code to use dynamic runtime also
// define the _DLL macro.
#if defined(_MSC_VER)
#pragma message(                                                               \
    "SYCL library is designed to work safely with dynamic C++ runtime."        \
    "Please use /MD switch with sycl.dll, /MDd switch with sycld.dll, "        \
    "or -fsycl switch to set C++ runtime automatically.")
#else
#warning "SYCL library is designed to work safely with dynamic C++ runtime."\
    "Please use /MD switch with sycl.dll, /MDd switch with sycld.dll, "\
    "or -fsycl switch to set C++ runtime automatically."
#endif
#endif // defined(_WIN32) && !defined(_DLL) && !defined(__SYCL_DEVICE_ONLY__)

// TODO deprecation attribute can not be applied to template alias. Find a way
// to show deprecation warning with those aliases anyway.

template <class T, class Alloc = std::allocator<T>>
using vector_class = std::vector<T, Alloc>;

using string_class
    __SYCL2020_DEPRECATED("use STL classes directly") = std::string;

template <class Sig> using function_class = std::function<Sig>;

using mutex_class
    __SYCL2020_DEPRECATED("use STL classes directly") = std::mutex;

template <class T, class Deleter = std::default_delete<T>>
using unique_ptr_class = std::unique_ptr<T, Deleter>;

template <class T> using shared_ptr_class = std::shared_ptr<T>;

template <class T> using weak_ptr_class = std::weak_ptr<T>;

template <class T> using hash_class = std::hash<T>;

using exception_ptr_class
    __SYCL2020_DEPRECATED("use STL classes directly") = std::exception_ptr;

template <typename T, typename... ArgsT>
std::unique_ptr<T> make_unique_ptr(ArgsT &&... Args) {
  return std::unique_ptr<T>(new T(std::forward<ArgsT>(Args)...));
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
