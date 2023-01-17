//==-- TargetFusionInfo.h - Encapsule target-specific fusion functionality -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_PASSES_TARGETFUSIONINFO_H
#define SYCL_FUSION_PASSES_TARGETFUSIONINFO_H

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

namespace llvm {

class TargetFusionInfoImpl {

public:
  virtual ~TargetFusionInfoImpl() = default;

  virtual void notifyFunctionsDelete(llvm::ArrayRef<Function *> Funcs) const {
    (void)Funcs;
  }

  virtual void addKernelFunction(Function *KernelFunc) const {
    (void)KernelFunc;
  }

  virtual void postProcessKernel(Function *KernelFunc) const {
    (void)KernelFunc;
  }

  virtual ArrayRef<StringRef> getKernelMetadataKeys() const { return {}; }

  virtual void createBarrierCall(IRBuilderBase &Builder,
                                 int BarrierFlags) const = 0;

  virtual unsigned getPrivateAddressSpace() const = 0;

  virtual unsigned getLocalAddressSpace() const = 0;

  virtual void updateAddressSpaceMetadata(Function *KernelFunc,
                                          ArrayRef<size_t> LocalSize,
                                          unsigned AddressSpace) const {
    (void)KernelFunc;
    (void)LocalSize;
  }

protected:
  explicit TargetFusionInfoImpl(llvm::Module *Mod) : LLVMMod{Mod} {};

  llvm::Module *LLVMMod;

  friend class TargetFusionInfo;
};

class SPIRVTargetFusionInfo : public TargetFusionInfoImpl {
public:
  void addKernelFunction(Function *KernelFunc) const override;

  ArrayRef<StringRef> getKernelMetadataKeys() const override;

  void postProcessKernel(Function *KernelFunc) const override;

  void createBarrierCall(IRBuilderBase &Builder,
                         int BarrierFlags) const override;

  // Corresponds to definition of spir_private and spir_local in
  // "clang/lib/Basic/Target/SPIR.h", "SPIRDefIsGenMap".
  unsigned getPrivateAddressSpace() const override { return 0; }
  unsigned getLocalAddressSpace() const override { return 3; }

  void updateAddressSpaceMetadata(Function *KernelFunc,
                                  ArrayRef<size_t> LocalSize,
                                  unsigned AddressSpace) const override;

private:
  using TargetFusionInfoImpl::TargetFusionInfoImpl;
};

class NVPTXTargetFusionInfo : public TargetFusionInfoImpl {
public:
  void notifyFunctionsDelete(llvm::ArrayRef<Function *> Funcs) const override;

  void addKernelFunction(Function *KernelFunc) const override;

  ArrayRef<StringRef> getKernelMetadataKeys() const override;

  void createBarrierCall(IRBuilderBase &Builder,
                         int BarrierFlags) const override;

  // Corresponds to the definitions in the LLVM NVPTX backend user guide:
  // https://llvm.org/docs/NVPTXUsage.html#address-spaces
  unsigned getPrivateAddressSpace() const override { return 0; }
  unsigned getLocalAddressSpace() const override { return 3; }

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
  void notifyFunctionsDelete(llvm::ArrayRef<Function *> Funcs) const {
    Impl->notifyFunctionsDelete(Funcs);
  }

  ///
  /// Notify the target-specific implementation that the function \p KernelFunc
  /// was added as a new kernel. This should be called AFTER the function has
  /// been added.
  void addKernelFunction(llvm::Function *KernelFunc) const {
    Impl->addKernelFunction(KernelFunc);
  }

  ///
  /// Target-specific post-processing of the new kernel function \p KernelFunc.
  /// This should be called AFTER the function has been added and defined.
  void postProcessKernel(Function *KernelFunc) const {
    Impl->postProcessKernel(KernelFunc);
  }

  ///
  /// Get the target-specific list of argument metadata attached to each
  /// function that should be collected and attached to the fused kernel.
  llvm::ArrayRef<llvm::StringRef> getKernelMetadataKeys() const {
    return Impl->getKernelMetadataKeys();
  }

  void createBarrierCall(IRBuilderBase &Builder, int BarrierFlags) const {
    Impl->createBarrierCall(Builder, BarrierFlags);
  }

  unsigned getPrivateAddressSpace() const {
    return Impl->getPrivateAddressSpace();
  }

  unsigned getLocalAddressSpace() const { return Impl->getLocalAddressSpace(); }

  void updateAddressSpaceMetadata(Function *KernelFunc,
                                  ArrayRef<size_t> LocalSize,
                                  unsigned AddressSpace) const {
    Impl->updateAddressSpaceMetadata(KernelFunc, LocalSize, AddressSpace);
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

#endif // SYCL_FUSION_PASSES_TARGETFUSIONINFO_H
