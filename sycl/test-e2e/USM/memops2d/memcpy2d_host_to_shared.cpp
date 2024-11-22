//==-- memcpy2d_host_to_shared.cpp - 2D memcpy from host USM to shared USM -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: aspect-usm_host_allocations, aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Temporarily disabled until the failure is addressed.
// UNSUPPORTED: (level_zero && windows)

// https://github.com/intel/llvm/issues/15648
// UNSUPPORTED: (gpu-intel-dg2 || hip_amd) && linux

#include "memcpy2d_common.hpp"

int main() { return test<Alloc::Host, Alloc::Shared>(); }
