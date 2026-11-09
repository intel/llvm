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
//
// The detection logic (Xe2 or newer = Intel GPU with IP version >= 0x05004000,
// i.e. 20.1.0, plus this driver version) is duplicated in the following places,
// which must be kept in sync with ur_device_handle_t_::isUsmPoolingDisabled():
// - sycl/test-e2e/lit.cfg.py (level_zero_usm_pooling_disabled feature),
// - unified-runtime/test/adapters/level_zero/enqueue_alloc.cpp
//   (isUsmPoolingDisabledForDevice()).
#define UR_L0_USM_POOLING_DISABLED_MIN_DRIVER_MAJOR 1
#define UR_L0_USM_POOLING_DISABLED_MIN_DRIVER_MINOR 17
