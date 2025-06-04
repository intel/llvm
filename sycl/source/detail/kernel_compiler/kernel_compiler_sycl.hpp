//==-- kernel_compiler_sycl.hpp --- SYCL kernel compilation support --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/export.hpp> // __SYCL_EXPORT
#include <sycl/detail/string_view.hpp>

#include <detail/compiler.hpp> // sycl_device_binaries

#include <string>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

using include_pairs_t = std::vector<std::pair<std::string, std::string>>;

std::string
userArgsAsString(const std::vector<sycl::detail::string_view> &UserArguments);

// Compile the given SYCL source string and virtual include files into the image
// format understood by the program manager.
//
// Returns a pointer to the image (owned by the `jit_compiler` class), and the
// bundle-specific prefix used for loading the kernels.
std::pair<sycl_device_binaries, std::string>
SYCL_JIT_Compile(const std::string &Source, const include_pairs_t &IncludePairs,
                 const std::vector<sycl::detail::string_view> &UserArgs,
                 std::string *LogPtr);

void SYCL_JIT_Destroy(sycl_device_binaries Binaries);

bool SYCL_JIT_Compilation_Available();

} // namespace detail
} // namespace ext::oneapi::experimental

} // namespace _V1
} // namespace sycl
