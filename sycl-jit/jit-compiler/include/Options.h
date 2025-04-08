//==-------- Options.h - Option infrastructure for the JIT compiler --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "DynArray.h"
#include "JITBinaryInfo.h"
#include "Macros.h"

namespace jit_compiler {

/// Unique ID for each supported architecture in the SYCL implementation.
///
/// Values of this type will only be used in the kernel fusion non-persistent
/// JIT. There is no guarantee for backwards compatibility, so this should not
/// be used in persistent caches.
using DeviceArchitecture = unsigned;

class TargetInfo {
public:
  static constexpr TargetInfo get(BinaryFormat Format,
                                  DeviceArchitecture Arch) {
    if (Format == BinaryFormat::SPIRV) {
      /// As an exception, SPIR-V targets have a single common ID (-1), as fused
      /// kernels will be reused across SPIR-V devices.
      return {Format, DeviceArchitecture(-1)};
    }
    return {Format, Arch};
  }

  TargetInfo() = default;

  constexpr BinaryFormat getFormat() const { return Format; }
  constexpr DeviceArchitecture getArch() const { return Arch; }

private:
  constexpr TargetInfo(BinaryFormat Format, DeviceArchitecture Arch)
      : Format(Format), Arch(Arch) {}

  BinaryFormat Format;
  DeviceArchitecture Arch;
};

enum OptionID {
  VerboseOutput,
  EnableCaching,
  TargetDeviceInfo,
  TargetCPU,
  TargetFeatures
};

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

struct JITTargetInfo
    : public OptionBase<JITTargetInfo, OptionID::TargetDeviceInfo, TargetInfo> {
  using OptionBase::OptionBase;
};

struct JITTargetCPU
    : public OptionBase<JITTargetCPU, OptionID::TargetCPU, DynArray<char>> {
  using OptionBase::OptionBase;
};

struct JITTargetFeatures
    : public OptionBase<JITTargetFeatures, OptionID::TargetFeatures,
                        DynArray<char>> {
  using OptionBase::OptionBase;
};

} // namespace option

extern "C" {

/// Clear all previously set options.
JIT_EXPORT_SYMBOL void resetJITConfiguration();

/// Add an option to the configuration.
JIT_EXPORT_SYMBOL void addToJITConfiguration(OptionStorage &&Opt);

} // extern "C"

} // namespace jit_compiler
