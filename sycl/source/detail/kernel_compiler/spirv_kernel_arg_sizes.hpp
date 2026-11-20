//==-- spirv_kernel_arg_sizes.hpp  Kernel argument sizes from SPIR-V ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

// Byte size the kernel expects for each of its parameters, in parameter order.
// An element is std::nullopt if the expected size cannot be determined
// reliably, e.g. for local memory pointers, images, samplers, or types whose
// layout the parser does not know.
using KernelArgSizesT = std::vector<std::optional<size_t>>;
using KernelNameToArgSizesMap =
    std::map<std::string, KernelArgSizesT, std::less<>>;

// Extracts the expected parameter sizes of all OpenCL kernel entry points in
// the SPIR-V module. This never throws on malformed input; it returns an empty
// map instead and leaves validation of the module to the backend.
KernelNameToArgSizesMap getSPIRVKernelArgSizes(const std::byte *Data,
                                               size_t Size);

} // namespace detail
} // namespace _V1
} // namespace sycl
