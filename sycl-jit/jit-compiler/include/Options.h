//===- Options.h - Option infrastructure for the JIT compiler -------------===//
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

/// Infrastructure to pass options to the SYCL-JIT library, for example:
/// ```
/// addToJITConfiguration(option::JITEnableVerbose::set(true))
/// ```

enum OptionID { VerboseOutput, TargetCPU, TargetFeatures };

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

  /// Create a suitable `OptionStorage` object with the given arguments.
  template <typename... Args> static OptionStorage set(Args &&...As) {
    return OptionStorage::makeOption<OptionT>(std::forward<Args>(As)...);
  }

  T Value;
};

namespace option {

/// Enable verbose output.
struct JITEnableVerbose
    : public OptionBase<JITEnableVerbose, OptionID::VerboseOutput, bool> {
  using OptionBase::OptionBase;
};

/// Set the target architecture to be used when JIT-ing kernels, e.g. SM version
/// for Nvidia.
struct JITTargetCPU
    : public OptionBase<JITTargetCPU, OptionID::TargetCPU, DynArray<char>> {
  using OptionBase::OptionBase;
};

/// Set the desired target features to be used when JIT-ing kernels, e.g. PTX
/// version for Nvidia.
struct JITTargetFeatures
    : public OptionBase<JITTargetFeatures, OptionID::TargetFeatures,
                        DynArray<char>> {
  using OptionBase::OptionBase;
};

} // namespace option

/// Clear all previously set options.
JIT_EXPORT_SYMBOL void resetJITConfiguration();

/// Add an option to the configuration.
JIT_EXPORT_SYMBOL void addToJITConfiguration(OptionStorage &&Opt);

} // namespace jit_compiler
