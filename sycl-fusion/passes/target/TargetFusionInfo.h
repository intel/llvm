//==-- TargetFusionInfo.h - Encapsule target-specific fusion functionality -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_PASSES_TARGET_TARGETFUSIONINFO_H
#define SYCL_FUSION_PASSES_TARGET_TARGETFUSIONINFO_H

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

namespace llvm {

class TargetFusionInfoImpl;

///
/// Common interface to target-specific logic around handling of kernel
/// functions.
class TargetFusionInfo {
public:
  ///
  /// Create the correct target-specific implementation based on the target
  /// triple of \p Module.
  explicit TargetFusionInfo(llvm::Module *Module);

  ///
  /// Notify the target-specific implementation that set of functions \p Funcs
  /// is about to be erased from the module. This should be called BEFORE
  /// erasing the functions.
  void notifyFunctionsDelete(llvm::ArrayRef<Function *> Funcs) const;

  ///
  /// Notify the target-specific implementation that the function \p KernelFunc
  /// was added as a new kernel. This should be called AFTER the function has
  /// been added.
  void addKernelFunction(llvm::Function *KernelFunc) const;

  ///
  /// Target-specific post-processing of the new kernel function \p KernelFunc.
  /// This should be called AFTER the function has been added and defined.
  void postProcessKernel(Function *KernelFunc) const;

  ///
  /// Get the target-specific list of argument metadata attached to each
  /// function that should be collected and attached to the fused kernel.
  llvm::ArrayRef<llvm::StringRef> getKernelMetadataKeys() const;

  ///
  /// Get the target-specific list of kernel function attributes that are
  /// uniform across all input kernels and should be attached to the fused
  /// kernel.
  llvm::ArrayRef<llvm::StringRef> getUniformKernelAttributes() const;

  void createBarrierCall(IRBuilderBase &Builder, int BarrierFlags) const;

  unsigned getPrivateAddressSpace() const;

  unsigned getLocalAddressSpace() const;

  void updateAddressSpaceMetadata(Function *KernelFunc,
                                  ArrayRef<size_t> LocalSize,
                                  unsigned AddressSpace) const;

private:
  using ImplPtr = std::shared_ptr<TargetFusionInfoImpl>;

  ImplPtr Impl;
};

///
/// Simple helper to collect a target-specific set of kernel argument metadata
/// from input functions and attach it to a fused kernel.
class MetadataCollection {
public:
  explicit MetadataCollection(llvm::ArrayRef<llvm::StringRef> MDKeys);

  void collectFromFunction(llvm::Function *Func,
                           const ArrayRef<bool> IsArgPresentMask);

  void attachToFunction(llvm::Function *Func);

private:
  llvm::SmallVector<llvm::StringRef> Keys;

  llvm::StringMap<llvm::SmallVector<llvm::Metadata *>> Collection;
};
} // namespace llvm

#endif // SYCL_FUSION_PASSES_TARGET_TARGETFUSIONINFO_H
