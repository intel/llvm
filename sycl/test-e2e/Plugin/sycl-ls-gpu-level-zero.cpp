// REQUIRES: gpu, level_zero

// RUN: sycl-ls --verbose >%t.default.out
// RUN: FileCheck %s --check-prefixes=CHECK-GPU-BUILTIN,CHECK-GPU-CUSTOM --input-file %t.default.out

// CHECK-GPU-BUILTIN: gpu_selector(){{.*}}gpu, {{.*}}Level-Zero
// CHECK-GPU-CUSTOM: custom_selector(gpu){{.*}}gpu, {{.*}}Level-Zero

//==-- sycl-ls-gpu-level-zero.cpp - Test Level-Zero selected gpu device ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Temporarily disable on L0 due to fails in CI
// UNSUPPORTED: level_zero
