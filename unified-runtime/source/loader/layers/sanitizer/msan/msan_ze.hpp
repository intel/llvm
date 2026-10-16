/*
 *
 * Copyright (C) 2026 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_ze.hpp
 *
 * MSan over the UR offload adapter: intercepts the Level Zero calls made by
 * liboffload's level_zero plugin through libur_msan_ze.so.
 */

#pragma once

#include "msan_ze_hooks.hpp"
#include "unified-runtime/ur_api.h"

namespace ur_sanitizer_layer {
namespace msan {

// Real L0 loader entry points; valid after loadZeShim() succeeds.
extern const MsanZeTable *ZeReal;

// MSan implementations of the hooked calls (msan_ze_intercept.cpp).
const MsanZeTable *getZeHooks();

// dlopen libur_msan_ze.so next to this layer and install the hooks. Must run
// before liboffload dlopens "libze_loader.so". Missing shim is not an error.
ur_result_t loadZeShim();

} // namespace msan
} // namespace ur_sanitizer_layer
