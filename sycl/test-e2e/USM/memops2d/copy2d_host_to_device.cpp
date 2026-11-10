//==---- copy2d_host_to_device.cpp - 2D copy from host USM to device USM ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: aspect-usm_host_allocations, aspect-usm_device_allocations
// RUN: %{build} -o %t.out
// TODO: revert this to a single RUN line once this is confirmed fixed on DG2.
// RUN: env UR_LOG_LEVEL_ZERO=level:info %{run} %t.out
// RUN: env UR_LOG_LEVEL_ZERO=level:info %{run} %t.out
// RUN: env UR_LOG_LEVEL_ZERO=level:info %{run} %t.out
// RUN: env UR_LOG_LEVEL_ZERO=level:info %{run} %t.out
// RUN: env UR_LOG_LEVEL_ZERO=level:info %{run} %t.out
// RUN: env UR_LOG_LEVEL_ZERO=level:info %{run} %t.out
// RUN: env UR_LOG_LEVEL_ZERO=level:info %{run} %t.out
// RUN: env UR_LOG_LEVEL_ZERO=level:info %{run} %t.out
// RUN: env UR_LOG_LEVEL_ZERO=level:info %{run} %t.out
// RUN: env UR_LOG_LEVEL_ZERO=level:info %{run} %t.out
// Experiment: does forcing V2 off immediate command lists avoid the bug?
// RUN: env UR_LOG_LEVEL_ZERO=level:info UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out
// RUN: env UR_LOG_LEVEL_ZERO=level:info UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out
// RUN: env UR_LOG_LEVEL_ZERO=level:info UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out
// RUN: env UR_LOG_LEVEL_ZERO=level:info UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out
// RUN: env UR_LOG_LEVEL_ZERO=level:info UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out
// RUN: env UR_LOG_LEVEL_ZERO=level:info UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out
// RUN: env UR_LOG_LEVEL_ZERO=level:info UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out
// RUN: env UR_LOG_LEVEL_ZERO=level:info UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out
// RUN: env UR_LOG_LEVEL_ZERO=level:info UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out
// RUN: env UR_LOG_LEVEL_ZERO=level:info UR_L0_V2_FORCE_BATCHED=1 %{run} %t.out %if gpu-intel-dg2 && level_zero_v2_adapter %{&& echo CI-VALIDATION-PASSED-BATCHED && false%}

// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/10157

#include "copy2d_common.hpp"

int main() { return test<Alloc::Host, Alloc::Device>(); }
