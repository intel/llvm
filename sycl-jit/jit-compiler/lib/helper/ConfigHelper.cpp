//===- ConfigHelper.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConfigHelper.h"

using namespace jit_compiler;

thread_local Config ConfigHelper::Cfg;

JIT_EXPORT_SYMBOL void resetJITConfiguration() { ConfigHelper::reset(); }

JIT_EXPORT_SYMBOL void addToJITConfiguration(OptionStorage &&Opt) {
  ConfigHelper::getConfig().set(std::move(Opt));
}
