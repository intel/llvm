//==---- copy2d_shared_to_host.cpp - 2D copy from shared USM to host USM ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// REQUIRES: aspect-usm_shared_allocations, aspect-usm_host_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: hip && linux
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/15648

#include "copy2d_common.hpp"

int main() { return test<Alloc::Shared, Alloc::Host>(); }
