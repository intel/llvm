//== memcpy2d_shared_to_device.cpp - 2D memcpy from shared USM to shared USM =//
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

#include "memcpy2d_common.hpp"

int main() { return test<Alloc::Shared, Alloc::Shared>(); }
