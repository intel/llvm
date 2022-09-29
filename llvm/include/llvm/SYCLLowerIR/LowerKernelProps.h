//===---- LowerKernelProps.h - lower kernel properties -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Lowers SYCL kernel properties into attributes used by sycl-post-link
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/IR/PassManager.h"

namespace llvm {

namespace sycl_kernel_props {
constexpr char ATTR_DOUBLE_GRF[] = "double-grf";
}

// Lowers calls to __sycl_set_kernel_properties
class SYCLLowerKernelPropsPass
    : public PassInfoMixin<SYCLLowerKernelPropsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // namespace llvm
