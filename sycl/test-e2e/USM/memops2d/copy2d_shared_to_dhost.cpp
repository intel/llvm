//==----- copy2d_shared_to_dhost.cpp - 2D copy from shared USM to host -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Temporarily disabled until the failure is addressed.
// UNSUPPORTED: gpu-intel-pvc || (level_zero && windows)

#include "copy2d_common.hpp"

int main() { return test<Alloc::Shared, Alloc::DirectHost>(); }
