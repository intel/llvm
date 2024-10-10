//==-------------------------- load_library.hpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Collection of helper OS-agnostic functions to dynamically load libraries and
// query their symbols.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <string>

namespace sycl {
inline namespace _V1 {
namespace detail {

// Function to load a shared library
// Implementation is OS dependent
void *loadOsLibrary(const std::string &Library);

// Function to unload a shared library
// Implementation is OS dependent (see posix-ur.cpp and windows-ur.cpp)
int unloadOsLibrary(void *Library);

// Function to get Address of a symbol defined in the shared
// library, implementation is OS dependent.
void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName);

} // namespace detail
} // namespace _V1
} // namespace sycl
