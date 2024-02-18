//==------- kernel_bundle_enums.hpp - SYCL kernel_bundle related enums -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

namespace sycl {
inline namespace _V1 {

enum class bundle_state : char {
  input = 0,
  object = 1,
  executable = 2,
  ext_oneapi_source = 3
};

namespace ext::oneapi::experimental {

enum class source_language : int { opencl = 0, spirv = 1 /* sycl, cuda */ };

} // namespace ext::oneapi::experimental

} // namespace _V1
} // namespace sycl
