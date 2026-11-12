//==---- copy2d_device_to_host.cpp - 2D copy from device USM to host USM ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: aspect-usm_device_allocations, aspect-usm_host_allocations
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
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched11.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched12.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched13.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched14.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched15.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched16.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched17.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched18.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched19.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched20.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched21.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched22.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched23.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched24.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched25.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched26.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched27.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched28.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched29.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out > %t.batched30.log 2>&1 || true
// Check the batched logs now, before the default path below (which can
// abort the script on a real failure) gets a chance to run.
// RUN: not grep -l "did not match" %t.batched1.log %t.batched2.log %t.batched3.log %t.batched4.log %t.batched5.log %t.batched6.log %t.batched7.log %t.batched8.log %t.batched9.log %t.batched10.log %t.batched11.log %t.batched12.log %t.batched13.log %t.batched14.log %t.batched15.log %t.batched16.log %t.batched17.log %t.batched18.log %t.batched19.log %t.batched20.log %t.batched21.log %t.batched22.log %t.batched23.log %t.batched24.log %t.batched25.log %t.batched26.log %t.batched27.log %t.batched28.log %t.batched29.log %t.batched30.log
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

// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/10157

#include "copy2d_common.hpp"

int main() { return test<Alloc::Device, Alloc::Host>(); }
