// RUN: sycl-ls --verbose | grep "Device \[" | wc -l >%t.verbose.out
// RUN: sycl-ls | wc -l >%t.concise.out
// RUN: diff %t.verbose.out %t.concise.out

//==---- sycl-ls.cpp - SYCL test for consistency of sycl-ls output ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The test crashed on CUDA CI machines with the latest OpenCL GPU RT
// (21.19.19792).
// UNSUPPORTED: cuda
// Temporarily disable on L0 due to fails in CI
// UNSUPPORTED: level_zero
