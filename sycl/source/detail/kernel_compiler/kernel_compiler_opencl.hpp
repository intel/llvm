//==-- kernel_compiler_opencl.hpp  OpenCL kernel compilation support       -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/export.hpp> // for __SYCL_EXPORT
#include <sycl/device.hpp>

#include <string>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

using spirv_vec_t = std::vector<uint8_t>;
spirv_vec_t OpenCLC_to_SPIRV(const std::string &Source,
                             const std::vector<uint32_t> &IPVersionVec,
                             const std::vector<std::string> &UserArgs,
                             std::string *LogPtr);

bool OpenCLC_Compilation_Available();

bool OpenCLC_Feature_Available(const std::string &Feature, uint32_t IPVersion);

bool OpenCLC_Supports_Version(
    const ext::oneapi::experimental::cl_version &Version, uint32_t IPVersion);

bool OpenCLC_Supports_Extension(
    const std::string &Name, ext::oneapi::experimental::cl_version *VersionPtr,
    uint32_t IPVersion);

std::string OpenCLC_Profile(uint32_t IPVersion);

} // namespace detail
} // namespace ext::oneapi::experimental

} // namespace _V1
} // namespace sycl
