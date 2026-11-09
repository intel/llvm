//===--- sanitizer_options.hpp - Device sanitizer runtime options ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

namespace llvm::offload::sanitizer {

// Only the options the MSan host runtime reads; unknown keys are ignored.
struct SanitizerOptions {
  bool Debug = false;
  bool Recover = false;
  bool MsanCheckHostAndSharedUSM = true;
};

// Parsed once from OFFLOAD_MSAN_OPTIONS.
const SanitizerOptions &getOptions();

} // namespace llvm::offload::sanitizer
