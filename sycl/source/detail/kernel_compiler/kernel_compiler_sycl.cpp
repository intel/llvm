//==-- kernel_compiler_opencl.cpp  OpenCL kernel compilation support       -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernel_compiler_sycl.hpp"
#include <sycl/exception.hpp> // make_error_code

#if SYCL_EXT_JIT_ENABLE
#include "../jit_compiler.hpp"
#endif

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

bool SYCL_JIT_Compilation_Available() {
#if SYCL_EXT_JIT_ENABLE
  return sycl::detail::jit_compiler::get_instance().isAvailable();
#else
  return false;
#endif
}

spirv_vec_t SYCL_JIT_to_SPIRV(
    [[maybe_unused]] const std::string &SYCLSource,
    [[maybe_unused]] include_pairs_t IncludePairs,
    [[maybe_unused]] const std::vector<std::string> &UserArgs,
    [[maybe_unused]] std::string *LogPtr,
    [[maybe_unused]] const std::vector<std::string> &RegisteredKernelNames) {
#if SYCL_EXT_JIT_ENABLE
  return sycl::detail::jit_compiler::get_instance().compileSYCL(
      "rtc", SYCLSource, IncludePairs, UserArgs, LogPtr, RegisteredKernelNames);
#else
  throw sycl::exception(sycl::errc::build,
                        "kernel_compiler via sycl-jit is not available");
#endif
}

std::string userArgsAsString(const std::vector<std::string> &UserArguments) {
  return std::accumulate(UserArguments.begin(), UserArguments.end(),
                         std::string(""),
                         [](const std::string &A, const std::string &B) {
                           return A.empty() ? B : A + " " + B;
                         });
}

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
