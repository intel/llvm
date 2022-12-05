//==---- ConfigHelper.h - Helper to manage compilation options for JIT -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_JIT_COMPILER_HELPER_CONFIGHELPER_H
#define SYCL_FUSION_JIT_COMPILER_HELPER_CONFIGHELPER_H

#include "Options.h"

namespace jit_compiler {

class ConfigHelper {
public:
  static void setConfig(Config &&JITConfig) { Cfg = std::move(JITConfig); }

  template <typename Opt> static typename Opt::ValueType get() {
    return Cfg.get<Opt>();
  }

private:
  static thread_local Config Cfg;
};
} // namespace jit_compiler

#endif // SYCL_FUSION_JIT_COMPILER_HELPER_CONFIGHELPER_H
