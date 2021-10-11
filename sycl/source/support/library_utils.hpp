//===-- library_utils.hpp - Dynamic library utilities ----------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
/// \returns a path to a directory from which plugins can be loaded. This can
/// be an empty string in unit tests only.
std::string getPluginDirectory();

/// Loads dynamic library.
///
/// \param Path is a path to dynamic library file.
/// \returns an OS-specific dynamic library handle
void *loadOsLibrary(const std::string &Path);

/// Unloads dynamic library.
///
/// \param Handle is a valid OS library handle, obtained from loadLibrary().
/// \returns 0 on success, error code otherwise.
int unloadOsLibrary(void *Handle);

/// Obtain a pointer to dynamic library symbol.
///
/// \param Handle is a valid library handle, returned from loadLibrary().
/// \param FunctionName is a symbol name.
/// \returns a valid function pointer if symbol is found, nullptr otherwise.
void *getOsLibraryFuncAddress(void *Handle, const std::string &FunctionName);
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
