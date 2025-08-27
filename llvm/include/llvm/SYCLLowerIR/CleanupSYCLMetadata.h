//===---------- CleanupSYCLMetadata.h - CleanupSYCLMetadata Pass ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Cleanup SYCL compiler internal metadata inserted by the frontend as it will
// never be used in the compilation ever again
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_CLEANUP_SYCL_METADATA
#define LLVM_CLEANUP_SYCL_METADATA

#include "llvm/IR/PassManager.h"

namespace llvm {

class CleanupSYCLMetadataPass : public PassInfoMixin<CleanupSYCLMetadataPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Removes the global variable "llvm.used".
/// "llvm.used" is a global constant array containing references to kernels
/// available in the module and callable from host code. The elements of
/// the array are ConstantExpr bitcast to i8*.
/// The variable must be removed because it has done the job to the moment
/// of a compilation stage and the references to the kernels callable from
/// host must not have users.
class CleanupSYCLMetadataFromLLVMUsed
    : public PassInfoMixin<CleanupSYCLMetadataFromLLVMUsed> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

/// Removes all device_global variables from the llvm.compiler.used global
/// variable. A device_global with internal linkage will be in
/// llvm.compiler.used to avoid the compiler wrongfully removing it during
/// optimizations. However, as an effect the device_global variables will also
/// be distributed across binaries, even if llvm.compiler.used has served its
/// purpose. To avoid polluting other binaries with unused device_global
/// variables, we remove them from llvm.compiler.used and erase them if they
/// have no further uses.
class RemoveDeviceGlobalFromLLVMCompilerUsed
    : public PassInfoMixin<RemoveDeviceGlobalFromLLVMCompilerUsed> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // namespace llvm

#endif // LLVM_CLEANUP_SYCL_METADATA
