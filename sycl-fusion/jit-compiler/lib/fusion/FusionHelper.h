//==--------- FusionHelper.h - Helpers to insert fused kernel stub ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_JIT_COMPILER_FUSION_FUSIONHELPER_H
#define SYCL_FUSION_JIT_COMPILER_FUSION_FUSIONHELPER_H

#include "Kernel.h"
#include "NDRangesHelper.h"
#include "Parameter.h"
#include "llvm/IR/Module.h"
#include <llvm/ADT/ArrayRef.h>
#include <vector>

namespace helper {

///
/// Simple helper to insert stubs and metadata for fused kernels into an LLVM
/// module.
class FusionHelper {
public:
  ///
  /// Representation of a fused kernel named FusedName and fusing
  /// all the kernels listed in FusedKernels.
  struct FusedFunction {
    FusedFunction(
        const char *FusedName, llvm::ArrayRef<std::string> FusedKernels,
        llvm::ArrayRef<jit_compiler::ParameterIdentity> ParameterIdentities,
        llvm::ArrayRef<jit_compiler::ParameterInternalization>
            ParameterInternalization,
        llvm::ArrayRef<jit_compiler::JITConstant> Constants,
        llvm::ArrayRef<jit_compiler::NDRange> NDRanges)
        : FusedName{FusedName}, FusedKernels{FusedKernels},
          ParameterIdentities{ParameterIdentities},
          ParameterInternalization{ParameterInternalization},
          Constants{Constants}, NDRanges{NDRanges},
          FusedNDRange{jit_compiler::combineNDRanges(NDRanges)} {}

    const char *FusedName;
    llvm::ArrayRef<std::string> FusedKernels;
    llvm::ArrayRef<jit_compiler::ParameterIdentity> ParameterIdentities;
    llvm::ArrayRef<jit_compiler::ParameterInternalization>
        ParameterInternalization;
    llvm::ArrayRef<jit_compiler::JITConstant> Constants;
    llvm::ArrayRef<jit_compiler::NDRange> NDRanges;
    jit_compiler::NDRange FusedNDRange;
  };

  ///
  /// Insert a function stub and metadata into the given LLVMModule,
  /// representing a fused kernel resulting from the fusion of all kernels
  /// listed in FusedFunctions. The actual fusion is performed by an LLVM pass,
  /// which will consume the metadata and extend the function stub accordingly.
  static llvm::Expected<std::unique_ptr<llvm::Module>>
  addFusedKernel(llvm::Module *LLVMModule,
                 const std::vector<FusedFunction> &FusedFunctions);

private:
  ///
  /// Create a fresh, "clean" module from LLVMMod, containing only the input
  /// functions from FusedFunctions and functions reachable from there.
  static llvm::Expected<std::unique_ptr<llvm::Module>>
  getCleanModule(llvm::Module *LLVMMod,
                 const std::vector<FusedFunction> &FusedFunctions);
};

} // namespace helper

#endif // SYCL_FUSION_JIT_COMPILER_FUSION_FUSIONHELPER_H
