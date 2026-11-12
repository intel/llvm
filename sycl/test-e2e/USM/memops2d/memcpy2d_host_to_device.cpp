//==-- memcpy2d_host_to_device.cpp - 2D memcpy from host USM to device USM -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: aspect-usm_host_allocations, aspect-usm_device_allocations
// RUN: %{build} -o %t.out
// TODO: revert this once #23392 is root-caused and fixed.
// Run the batched (non-immediate command list) experiment first and
// unconditionally, capturing its output, so a real failure in the default
// path below can't prevent us from seeing whether batched mode avoids it.
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched1.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched2.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched3.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched4.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched5.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched6.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched7.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched8.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched9.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched10.log 2>&1 || true
// Default (immediate command list) path; repeated for intermittent
// failures. A real failure here aborts the test, as before.
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// RUN: %{run} %t.out
// Surface a batched-only failure even if the default path above passed.
// RUN: not grep -q "did not match" %t.batched1.log %t.batched2.log %t.batched3.log %t.batched4.log %t.batched5.log %t.batched6.log %t.batched7.log %t.batched8.log %t.batched9.log %t.batched10.log

// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/10157

#include "memcpy2d_common.hpp"

int main() { return test<Alloc::Host, Alloc::Device>(); }
