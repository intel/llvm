//==------------ ModuleInfo.h - Manages kernel info for the JIT ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_COMMON_MODULEINFO_H
#define SYCL_FUSION_COMMON_MODULEINFO_H

#include "Kernel.h"

#include <algorithm>
#include <string>
#include <vector>

namespace jit_compiler {

///
/// Represents a SPIR-V translation unit containing SYCL kernels by the
/// KernelInfo for each of the contained kernels.
class SYCLModuleInfo {
public:
  using KernelInfoList = std::vector<SYCLKernelInfo>;

  void addKernel(SYCLKernelInfo &Kernel) { Kernels.push_back(Kernel); }

  KernelInfoList &kernels() { return Kernels; }

  bool hasKernelFor(const std::string &KernelName) {
    return getKernelFor(KernelName) != nullptr;
  }

  SYCLKernelInfo *getKernelFor(const std::string &KernelName) {
    auto It =
        std::find_if(Kernels.begin(), Kernels.end(), [&](SYCLKernelInfo &K) {
          return K.Name == KernelName.c_str();
        });
    return (It != Kernels.end()) ? &*It : nullptr;
  }

private:
  KernelInfoList Kernels;
};

} // namespace jit_compiler

#endif // SYCL_FUSION_COMMON_MODULEINFO_H
