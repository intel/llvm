//===- ConfigHelper.h - Helper to manage compilation options for JIT ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Options.h"

#include <memory>
#include <unordered_map>

namespace jit_compiler {

class Config {

public:
  template <typename Opt> typename Opt::ValueType get() const {
    using T = typename Opt::ValueType;

    auto *ConfigValue = get(Opt::Id);
    if (!ConfigValue) {
      return T{};
    }
    return static_cast<const OptionBase<Opt, Opt::Id, T> *>(ConfigValue)->Value;
  }

  void set(OptionStorage &&Option) {
    auto ID = Option.get()->Id;
    OptionValues[ID] = std::move(Option);
  }

private:
  std::unordered_map<OptionID, OptionStorage> OptionValues;

  const OptionPtrBase *get(OptionID ID) const {
    const auto Iter = OptionValues.find(ID);
    if (Iter == OptionValues.end()) {
      return nullptr;
    }
    return Iter->second.get();
  }
};

class ConfigHelper {
public:
  static void reset() { Cfg = {}; }
  static Config &getConfig() { return Cfg; }

  template <typename Opt> static typename Opt::ValueType get() {
    return Cfg.get<Opt>();
  }

private:
  static thread_local Config Cfg;
};
} // namespace jit_compiler
