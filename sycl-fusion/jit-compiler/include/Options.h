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

#include <memory>
#include <unordered_map>

namespace jit_compiler {

enum OptionID { VerboseOutput, EnableCaching, TargetDeviceInfo };

class OptionPtrBase {};

class Config {

public:
  template <typename Opt> void set(typename Opt::ValueType Value) {
    Opt::set(*this, Value);
  }

  template <typename Opt> typename Opt::ValueType get() {
    return Opt::get(*this);
  }

private:
  std::unordered_map<OptionID, std::unique_ptr<OptionPtrBase>> OptionValues;

  void set(OptionID ID, std::unique_ptr<OptionPtrBase> Value) {
    OptionValues[ID] = std::move(Value);
  }

  OptionPtrBase *get(OptionID ID) {
    if (OptionValues.count(ID)) {
      return OptionValues.at(ID).get();
    }
    return nullptr;
  }

  template <OptionID ID, typename T> friend class OptionBase;
};

template <OptionID ID, typename T> class OptionBase : public OptionPtrBase {
public:
  using ValueType = T;

protected:
  static void set(Config &Cfg, T Value) {
    Cfg.set(ID,
            std::unique_ptr<OptionBase<ID, T>>{new OptionBase<ID, T>{Value}});
  }

  static const T get(Config &Cfg) {
    auto *ConfigValue = Cfg.get(ID);
    if (!ConfigValue) {
      return T{};
    }
    return static_cast<OptionBase<ID, T> *>(ConfigValue)->Value;
  }

private:
  T Value;

  OptionBase(T Val) : Value{Val} {}

  friend Config;
};

namespace option {

struct JITEnableVerbose : public OptionBase<OptionID::VerboseOutput, bool> {};

struct JITEnableCaching : public OptionBase<OptionID::EnableCaching, bool> {};

struct JITTargetInfo
    : public OptionBase<OptionID::TargetDeviceInfo, TargetInfo> {};

} // namespace option
} // namespace jit_compiler

#endif // SYCL_FUSION_JIT_COMPILER_OPTIONS_H
