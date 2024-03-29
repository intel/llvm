//==--- FusionPipeline - LLVM pass pipeline definition for kernel fusion ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_JIT_COMPILER_FUSION_FUSIONPIPELINE_H
#define SYCL_FUSION_JIT_COMPILER_FUSION_FUSIONPIPELINE_H

#include "Kernel.h"
#include "ModuleInfo.h"
#include "llvm/IR/Module.h"

namespace jit_compiler {
namespace fusion {

class FusionPipeline {
public:
  ///
  /// Run the necessary passes in a custom pass pipeline to perform kernel
  /// fusion on the given module. The module should contain the stub functions
  /// and fusion metadata. The given SYCLModuleInfo must contain information
  /// about all input kernels. The returned SYCLModuleInfo will additionally
  /// contain an entry for the fused kernel.
  static std::unique_ptr<SYCLModuleInfo>
  runFusionPasses(llvm::Module &Mod, SYCLModuleInfo &InputInfo,
                  BarrierFlags BarriersFlags);
};
} // namespace fusion
} // namespace jit_compiler

#endif // SYCL_FUSION_JIT_COMPILER_FUSION_FUSIONPIPELINE_H
