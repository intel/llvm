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
  virtual ~OptionPtrBase() = default;

  const OptionID Id;
};

class OptionStorage {
public:
  ~OptionStorage() { delete Storage; }

  OptionStorage() : Storage{nullptr} {}

  OptionStorage(const OptionStorage &) = delete;
  OptionStorage &operator=(const OptionStorage &) = delete;

  OptionStorage(OptionStorage &&Other) : Storage{Other.Storage} {
    Other.Storage = nullptr;
  }

  OptionStorage &operator=(OptionStorage &&Other) {
    Storage = Other.Storage;
    Other.Storage = nullptr;
    return *this;
  }

  OptionPtrBase *get() const { return Storage; }

  template <typename OptionT, typename... Args>
  static OptionStorage makeOption(Args &&...As) {
    return OptionStorage(new OptionT(std::forward<Args>(As)...));
  }

private:
  OptionPtrBase *Storage;

  OptionStorage(OptionPtrBase *Store) : Storage{Store} {}
};

template <typename OptionT, OptionID ID, typename T>
struct OptionBase : public OptionPtrBase {
  static constexpr OptionID Id = ID;
  using ValueType = T;

  template <typename... Args>
  explicit OptionBase(Args &&...As)
      : OptionPtrBase{ID}, Value{std::forward<Args>(As)...} {}

  template <typename... Args> static OptionStorage set(Args &&...As) {
    return OptionStorage::makeOption<OptionT>(std::forward<Args>(As)...);
  }

  T Value;
};

namespace option {

struct JITEnableVerbose
    : public OptionBase<JITEnableVerbose, OptionID::VerboseOutput, bool> {
  using OptionBase::OptionBase;
};

struct JITEnableCaching
    : public OptionBase<JITEnableCaching, OptionID::EnableCaching, bool> {
  using OptionBase::OptionBase;
};

struct JITTargetInfo
    : public OptionBase<JITTargetInfo, OptionID::TargetDeviceInfo, TargetInfo> {
  using OptionBase::OptionBase;
};

} // namespace option
} // namespace jit_compiler

#endif // SYCL_FUSION_JIT_COMPILER_OPTIONS_H
