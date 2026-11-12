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
// Isolate the copy-offload-hint variable from the submission-mode
// variable: keep immediate command lists (the default), but force the
// driver to disable copy engine offload globally, independent of any
// DG2-specific gating in this adapter's own code.
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled1.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled2.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled3.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled4.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled5.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled6.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled7.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled8.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled9.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled10.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled11.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled12.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled13.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled14.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled15.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled16.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled17.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled18.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled19.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled20.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled21.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled22.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled23.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled24.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled25.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled26.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled27.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled28.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled29.log 2>&1 || true
// RUN: env UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 %{run} %t.out > %t.offload_disabled30.log 2>&1 || true
// RUN: not grep -l "did not match" %t.offload_disabled1.log %t.offload_disabled2.log %t.offload_disabled3.log %t.offload_disabled4.log %t.offload_disabled5.log %t.offload_disabled6.log %t.offload_disabled7.log %t.offload_disabled8.log %t.offload_disabled9.log %t.offload_disabled10.log %t.offload_disabled11.log %t.offload_disabled12.log %t.offload_disabled13.log %t.offload_disabled14.log %t.offload_disabled15.log %t.offload_disabled16.log %t.offload_disabled17.log %t.offload_disabled18.log %t.offload_disabled19.log %t.offload_disabled20.log %t.offload_disabled21.log %t.offload_disabled22.log %t.offload_disabled23.log %t.offload_disabled24.log %t.offload_disabled25.log %t.offload_disabled26.log %t.offload_disabled27.log %t.offload_disabled28.log %t.offload_disabled29.log %t.offload_disabled30.log
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
