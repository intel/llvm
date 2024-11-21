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
// RUN: %{run} %t.out

// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/10157

// Temporarily disabled until the failure is addressed.
// UNSUPPORTED: level_zero && windows

#include "memcpy2d_common.hpp"

int main() { return test<Alloc::Host, Alloc::Device>(); }
