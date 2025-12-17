//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

namespace jit_compiler::resource {
// `resource.cpp` is compiled using freshly built clang and it's very hard to
// sync compilation options between that and normal compilation for other files.
// Note that some of the options might affect ABI (e.g., libstdc++ vs. libc++
// usage, or custom sysroot/gcc installation directory). A much easier approach
// is to ensure that `resource.cpp` doesn't have any includes at all, hence
// these helpers:
template <class T, unsigned long long N>
constexpr unsigned long long size(const T (&)[N]) noexcept {
  return N;
}
struct resource_string_view {
  template <unsigned long long N>
  resource_string_view(const char (&S)[N]) : S(S), Size(N - 1) {}
  const char *S;
  unsigned long long Size;
};
struct resource_file {
  resource_string_view Path;
  resource_string_view Content;
};
// Defined in the auto-generated file:
extern const resource_file ToolchainFiles[];
extern unsigned long long NumToolchainFiles;
extern resource_string_view ToolchainPrefix;
} // namespace jit_compiler::resource
