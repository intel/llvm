//===--- NativeCPU.h - Declare NativeCPU target feature support -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares NativeCPU TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_NATIVECPU_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_NATIVECPU_H

#include "Targets.h"

namespace clang {
namespace targets {

class LLVM_LIBRARY_VISIBILITY NativeCPUTargetInfo final : public TargetInfo {
  std::unique_ptr<TargetInfo> HostTarget;

public:
  NativeCPUTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts);

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    DefineStd(Builder, "NativeCPU", Opts);
  }

  SmallVector<Builtin::InfosShard> getTargetBuiltins() const override {
    return {};
  }

  BuiltinVaListKind getBuiltinVaListKind() const override {
    if (HostTarget)
      return HostTarget->getBuiltinVaListKind();

    return TargetInfo::VoidPtrBuiltinVaList;
  }

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &info) const override {
    return true;
  }

  std::string_view getClobbers() const override { return ""; }

  void setSupportedOpenCLOpts() override { supportAllOpenCLOpts(); }

  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override {
    if (HostTarget)
      return HostTarget->checkCallingConvention(CC);

    return TargetInfo::checkCallingConvention(CC);
  }

protected:
  void setAuxTarget(const TargetInfo *Aux) override;

  ArrayRef<const char *> getGCCRegNames() const override { return {}; }

  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    return {};
  }

  bool hasBitIntType() const override { return true; }
};

} // namespace targets
} // namespace clang

#endif // LLVM_CLANG_LIB_BASIC_TARGETS_NATIVECPU_H
