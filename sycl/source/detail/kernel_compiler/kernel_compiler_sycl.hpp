//==-- kernel_compiler_sycl.hpp   SYCL kernel compilation support          -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/export.hpp> // __SYCL_EXPORT
#include <sycl/device.hpp>

#include <detail/compiler.hpp> // sycl_device_binaries

#include <numeric> // std::accumulate
#include <string>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

using spirv_vec_t = std::vector<uint8_t>;
using include_pairs_t = std::vector<std::pair<std::string, std::string>>;

spirv_vec_t
SYCL_to_SPIRV(const std::string &Source, include_pairs_t IncludePairs,
              const std::vector<std::string> &UserArgs, std::string *LogPtr,
              const std::vector<std::string> &RegisteredKernelNames);

bool SYCL_Compilation_Available();

std::string userArgsAsString(const std::vector<std::string> &UserArguments);

// Compile the given SYCL source string and virtual include files into the image
// format understood by the program manager.
//
// Returns a pointer to the image (owned by the `jit_compiler` class), and the
// bundle-specific prefix used for loading the kernels.
std::pair<sycl_device_binaries, std::string> SYCL_JIT_to_SPIRV(
    const std::string &Source, const include_pairs_t &IncludePairs,
    const std::vector<std::string> &UserArgs, std::string *LogPtr);

void SYCL_JIT_destroy(sycl_device_binaries Binaries);

bool SYCL_JIT_Compilation_Available();

} // namespace detail
} // namespace ext::oneapi::experimental

} // namespace _V1
} // namespace sycl
