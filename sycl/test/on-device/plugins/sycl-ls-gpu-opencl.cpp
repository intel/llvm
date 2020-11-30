// REQUIRES: gpu, opencl

// RUN: %BE_RUN_PLACEHOLDER sycl-ls --verbose >%t.opencl.out
// RUN: FileCheck %s --check-prefixes=CHECK-GPU-BUILTIN,CHECK-GPU-CUSTOM --input-file %t.opencl.out

// CHECK-GPU-BUILTIN: gpu_selector(){{.*}}GPU : {{.*}}OpenCL
// CHECK-GPU-CUSTOM: custom_selector(gpu){{.*}}GPU : {{.*}}OpenCL

//==-- sycl-ls-gpu-opencl.cpp - SYCL test for selected OpenCL GPU device --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
