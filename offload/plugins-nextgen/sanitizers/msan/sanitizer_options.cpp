//===--- sanitizer_options.cpp - Device sanitizer runtime options ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sanitizer_options.hpp"
#include "sanitizer_log.hpp"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include <cstdlib>
#include <tuple>

namespace llvm::offload::sanitizer {

namespace {

// Same syntax as unified-runtime's getenv_to_map: "k1:v1,v2;k2:v3". Only the
// first value of a key is used.
SanitizerOptions parseOptions(const char *EnvName) {
  SanitizerOptions Opts;
  const char *Env = std::getenv(EnvName);
  if (!Env)
    return Opts;

  StringRef Rest(Env);
  while (!Rest.empty()) {
    StringRef Entry;
    std::tie(Entry, Rest) = Rest.split(';');
    if (Entry.trim().empty())
      continue;
    auto [Key, Values] = Entry.split(':');
    Key = Key.trim();
    StringRef Value = Values.split(',').first.trim();

    bool *Field = StringSwitch<bool *>(Key)
                      .Case("debug", &Opts.Debug)
                      .Case("recover", &Opts.Recover)
                      .Case("msan_check_host_and_shared_usm",
                            &Opts.MsanCheckHostAndSharedUSM)
                      .Default(nullptr);
    if (!Field)
      continue;
    if (Value.equals_insensitive("1") || Value.equals_insensitive("true")) {
      *Field = true;
    } else if (Value.equals_insensitive("0") ||
               Value.equals_insensitive("false")) {
      *Field = false;
    } else {
      SANITIZER_REPORT("<SANITIZER>[ERROR]: \"{}\" is set to \"{}\", which is "
                       "not a valid setting. Acceptable input are: for "
                       "enable, use: \"1\" \"true\"; for disable, use: \"0\" "
                       "\"false\".",
                       Key, Value);
      SANITIZER_REPORT("Sanitizer failed to parse options.");
      std::exit(1);
    }
  }
  return Opts;
}

} // namespace

const SanitizerOptions &getOptions() {
  // Fall back to the UR layer's variable, which SYCL users and tests set.
  static const SanitizerOptions Opts = parseOptions(
      std::getenv("OFFLOAD_MSAN_OPTIONS") ? "OFFLOAD_MSAN_OPTIONS"
                                          : "UR_LAYER_MSAN_OPTIONS");
  return Opts;
}

} // namespace llvm::offload::sanitizer
