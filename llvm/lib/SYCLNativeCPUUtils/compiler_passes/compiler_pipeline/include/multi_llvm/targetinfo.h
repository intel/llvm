// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/uxlfoundation/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MULTI_LLVM_TARGET_TARGETINFO_H_INCLUDED
#define MULTI_LLVM_TARGET_TARGETINFO_H_INCLUDED

#include <clang/Basic/TargetInfo.h>
#include <multi_llvm/llvm_version.h>

namespace multi_llvm {

namespace detail {

#if LLVM_VERSION_GREATER_EQUAL(21, 0)

template <typename TargetInfo = clang::TargetInfo>
auto createTargetInfo(clang::DiagnosticsEngine &Diags,
                      clang::TargetOptions &Opts)
    -> decltype(TargetInfo::CreateTargetInfo(Diags, Opts)) {
  return TargetInfo::CreateTargetInfo(Diags, Opts);
}

#endif

template <typename TargetInfo = clang::TargetInfo>
auto createTargetInfo(clang::DiagnosticsEngine &Diags,
                      clang::TargetOptions &Opts)
    -> decltype(TargetInfo::CreateTargetInfo(
        Diags, std::make_shared<clang::TargetOptions>(Opts))) {
  return TargetInfo::CreateTargetInfo(
      Diags, std::make_shared<clang::TargetOptions>(Opts));
}

}  // namespace detail

struct TargetInfo {
  static clang::TargetInfo *CreateTargetInfo(clang::DiagnosticsEngine &Diags,
                                             clang::TargetOptions &Opts) {
    return multi_llvm::detail::createTargetInfo(Diags, Opts);
  }
};

}  // namespace multi_llvm

#endif  // MULTI_LLVM_TARGET_TARGETINFO_H_INCLUDED
