//==--- cpu.cpp - AOT compilation for cpu devices using opencl-aot --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

// REQUIRES: opencl-aot, cpu

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 %S/Inputs/aot.cpp -o %t.out
// RUN: %{run} %t.out

// Test that opencl-aot can handle multiple build options.
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64  %S/Inputs/aot.cpp -Xsycl-target-backend "--bo=-g" -Xsycl-target-backend "--bo=-cl-opt-disable" -o %t2.out

// Test that opencl-aot can handle march option.
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64  %S/Inputs/aot.cpp -Xsycl-target-backend "--march=avx512"
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64  %S/Inputs/aot.cpp -Xsycl-target-backend "--march=wsm"
