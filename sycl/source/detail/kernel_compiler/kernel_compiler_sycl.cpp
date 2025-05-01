//==-- kernel_compiler_sycl.cpp --- SYCL kernel compilation support --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernel_compiler_sycl.hpp"
#include <sycl/exception.hpp> // make_error_code

#include <atomic>
#include <numeric> // std::accumulate

#if SYCL_EXT_JIT_ENABLE
#include <detail/jit_compiler.hpp>
#endif

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

std::string
userArgsAsString(const std::vector<sycl::detail::string_view> &UserArguments) {
  std::string Result = "";
  for (const sycl::detail::string_view &UserArg : UserArguments) {
    if (!Result.empty())
      Result += " ";
    Result += UserArg.data();
  }
  return Result;
}

bool SYCL_JIT_Compilation_Available() {
#if SYCL_EXT_JIT_ENABLE
  return sycl::detail::jit_compiler::get_instance().isAvailable();
#else
  return false;
#endif
}

std::pair<sycl_device_binaries, std::string> SYCL_JIT_Compile(
    [[maybe_unused]] const std::string &SYCLSource,
    [[maybe_unused]] const include_pairs_t &IncludePairs,
    [[maybe_unused]] const std::vector<sycl::detail::string_view> &UserArgs,
    [[maybe_unused]] std::string *LogPtr) {
#if SYCL_EXT_JIT_ENABLE
  static std::atomic_uintptr_t CompilationCounter;
  std::string CompilationID = "rtc_" + std::to_string(CompilationCounter++);
  std::vector<std::string> UserArgStrings;
  for (const sycl::detail::string_view UserArg : UserArgs)
    UserArgStrings.push_back(UserArg.data());
  return sycl::detail::jit_compiler::get_instance().compileSYCL(
      CompilationID, SYCLSource, IncludePairs, UserArgStrings, LogPtr);
#else
  throw sycl::exception(sycl::errc::build,
                        "kernel_compiler via sycl-jit is not available");
#endif
}

void SYCL_JIT_Destroy([[maybe_unused]] sycl_device_binaries Binaries) {
#if SYCL_EXT_JIT_ENABLE
  sycl::detail::jit_compiler::get_instance().destroyDeviceBinaries(Binaries);
#else
  throw sycl::exception(sycl::errc::invalid,
                        "kernel_compiler via sycl-jit is not available");
#endif
}

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
