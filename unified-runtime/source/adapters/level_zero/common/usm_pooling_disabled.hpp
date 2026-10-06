//===--------- usm_pooling_disabled.hpp - Level Zero Adapter -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

// Minimum L0 driver version (major.minor) starting from which USM pooling
// in the adapter is disabled on Xe2 or newer devices.
#define UR_L0_USM_POOLING_DISABLED_MIN_DRIVER_MAJOR 1
#define UR_L0_USM_POOLING_DISABLED_MIN_DRIVER_MINOR 17
