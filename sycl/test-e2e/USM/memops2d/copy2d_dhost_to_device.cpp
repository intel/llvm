//==----- copy2d_dhost_to_device.cpp - 2D copy from host to device USM -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: aspect-usm_device_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Temporarily disabled until the failure is addressed.
// UNSUPPORTED: (level_zero && windows)

#include "copy2d_common.hpp"

int main() { return test<Alloc::DirectHost, Alloc::Device>(); }
