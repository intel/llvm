//===--- msan_ze.hpp - MSan over the Level Zero plugin --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Intercepts the Level Zero calls made by the level_zero plugin through
// liboffload_msan_ze.so.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "msan_ze_hooks.hpp"

namespace llvm::offload::sanitizer {
namespace msan {

// Real L0 loader entry points; valid after loadZeShim() succeeds.
extern const MsanZeTable *ZeReal;

// MSan implementations of the hooked calls (msan_ze_intercept.cpp).
const MsanZeTable *getZeHooks();

// dlopen liboffload_msan_ze.so next to this library and install the hooks.
// Must run before the plugin dlopens "libze_loader.so". Returns false if the
// shim is missing.
bool loadZeShim();

} // namespace msan
} // namespace llvm::offload::sanitizer
