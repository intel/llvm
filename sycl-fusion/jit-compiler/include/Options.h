//==-------- Options.h - Option infrastructure for the JIT compiler --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_JIT_COMPILER_OPTIONS_H
#define SYCL_FUSION_JIT_COMPILER_OPTIONS_H

#include "Kernel.h"

namespace jit_compiler {

enum OptionID { VerboseOutput, EnableCaching, TargetDeviceInfo };

class OptionPtrBase {
protected:
  explicit OptionPtrBase(OptionID Id) : Id(Id) {}

public:
  const OptionID Id;
};

template <OptionID ID, typename T> struct OptionBase : public OptionPtrBase {
  static constexpr OptionID Id = ID;
  using ValueType = T;

  template <typename... Args>
  explicit OptionBase(Args &&...As)
      : OptionPtrBase{ID}, Value{std::forward<Args>(As)...} {}

  T Value;
};

namespace option {

struct JITEnableVerbose : public OptionBase<OptionID::VerboseOutput, bool> {
  using OptionBase::OptionBase;
};

struct JITEnableCaching : public OptionBase<OptionID::EnableCaching, bool> {
  using OptionBase::OptionBase;
};

struct JITTargetInfo
    : public OptionBase<OptionID::TargetDeviceInfo, TargetInfo> {
  using OptionBase::OptionBase;
};

} // namespace option
} // namespace jit_compiler

#endif // SYCL_FUSION_JIT_COMPILER_OPTIONS_H
