//==-- TargetFusionInfo.h - Encapsule target-specific fusion functionality -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

namespace llvm {

class TargetFusionInfoImpl {

public:
  virtual ~TargetFusionInfoImpl() = default;

  virtual void notifyFunctionsDelete(llvm::ArrayRef<Function *> Funcs) {
    (void)Funcs;
  }

  virtual void addKernelFunction(Function *KernelFunc) { (void)KernelFunc; }

  virtual void postProcessKernel(Function *KernelFunc) { (void)KernelFunc; }

  virtual ArrayRef<StringRef> getKernelMetadataKeys() { return {}; }

  virtual void createBarrierCall(IRBuilderBase &Builder, int BarrierFlags) = 0;

protected:
  explicit TargetFusionInfoImpl(llvm::Module *Mod) : LLVMMod{Mod} {};

  llvm::Module *LLVMMod;

  friend class TargetFusionInfo;
};

class SPIRVTargetFusionInfo : public TargetFusionInfoImpl {
public:
  void addKernelFunction(Function *KernelFunc) override;

  ArrayRef<StringRef> getKernelMetadataKeys() override;

  void postProcessKernel(Function *KernelFunc) override;

  void createBarrierCall(IRBuilderBase& Builder, int BarrierFlags) override;

private:
  using TargetFusionInfoImpl::TargetFusionInfoImpl;
};

class NVPTXTargetFusionInfo : public TargetFusionInfoImpl {
public:
  void notifyFunctionsDelete(llvm::ArrayRef<Function *> Funcs) override;

  void addKernelFunction(Function *KernelFunc) override;

  ArrayRef<StringRef> getKernelMetadataKeys() override;

  void createBarrierCall(IRBuilderBase &Builder, int BarrierFlags) override;

private:
  using TargetFusionInfoImpl::TargetFusionInfoImpl;
};

///
/// Common interface to target-specific logic around handling of kernel
/// functions.
class TargetFusionInfo {
public:
  ///
  /// Create the correct target-specific implementation based on the target
  /// triple of \p Module.
  static TargetFusionInfo getTargetFusionInfo(llvm::Module *Module);

  ///
  /// Notify the target-specific implementation that set of functions \p Funcs
  /// is about to be erased from the module. This should be called BEFORE
  /// erasing the functions.
  void notifyFunctionsDelete(llvm::ArrayRef<Function *> Funcs) {
    Impl->notifyFunctionsDelete(Funcs);
  }

  ///
  /// Notify the target-specific implementation that the function \p KernelFunc
  /// was added as a new kernel. This should be called AFTER the function has
  /// been added.
  void addKernelFunction(llvm::Function *KernelFunc) {
    Impl->addKernelFunction(KernelFunc);
  }

  ///
  /// Target-specific post-processing of the new kernel function \p KernelFunc.
  /// This should be called AFTER the function has been added and defined.
  void postProcessKernel(Function *KernelFunc) {
    Impl->postProcessKernel(KernelFunc);
  }

  ///
  /// Get the target-specific list of argument metadata attached to each
  /// function that should be collected and attached to the fused kernel.
  llvm::ArrayRef<llvm::StringRef> getKernelMetadataKeys() {
    return Impl->getKernelMetadataKeys();
  }

  void createBarrierCall(IRBuilderBase &Builder, int BarrierFlags) {
    Impl->createBarrierCall(Builder, BarrierFlags);
  }

private:
  using ImplPtr = std::shared_ptr<TargetFusionInfoImpl>;

  TargetFusionInfo(ImplPtr &&I) : Impl{I} {}

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
