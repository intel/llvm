//==----- cpu.cpp - AOT compilation for cpu devices using opencl-aot  --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------===//

// REQUIRES: opencl-aot, cpu

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice %S/Inputs/aot.cpp -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
