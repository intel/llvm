//==------ bitreverse_wingen12.cpp - DPC++ ESIMD bitreverse test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: windows && gpu-intel-gen12
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Win Gen12 variant of the test, without the driver version requirement as the
// Gen12 driver is no longer updated on Windows.

#include "bitreverse.cpp"
