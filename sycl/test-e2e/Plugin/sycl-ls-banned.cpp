// REQUIRES: cuda

// RUN: sycl-ls --verbose >%t.cuda.out
// RUN: FileCheck %s --input-file %t.cuda.out

//==---- sycl-ls-banned.cpp - Check sycl-ls output of banned platforms. --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// CHECK: Unsupported Platforms:
// CHECK-NEXT: Platform [#1]:
// CHECK-NEXT: Version  : OpenCL
