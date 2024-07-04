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

} // namespace detail
} // namespace ext::oneapi::experimental

} // namespace _V1
} // namespace sycl
