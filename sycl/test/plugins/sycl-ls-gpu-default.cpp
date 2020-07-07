// REQUIRES: gpu, level0

// RUN: sycl-ls --verbose >%t.default.out
// RUN: FileCheck %s --check-prefixes=CHECK-GPU-BUILTIN,CHECK-GPU-CUSTOM --input-file %t.default.out

// CHECK-GPU-BUILTIN: gpu_selector(){{.*}}GPU : 0.91
// CHECK-GPU-CUSTOM: custom_selector(gpu){{.*}}GPU : 0.91

//==-- sycl-ls-gpu-default.cpp - SYCL test for default selected gpu device -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
