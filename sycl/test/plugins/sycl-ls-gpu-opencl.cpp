// REQUIRES: gpu, opencl

// RUN: sycl-ls --verbose >%t.default.out
// RUN: FileCheck %s --check-prefixes=CHECK-GPU-BUILTIN,CHECK-GPU-CUSTOM --input-file %t.default.out

// RUN: env SYCL_BE=PI_OPENCL sycl-ls --verbose >%t.opencl.out
// RUN: FileCheck %s --check-prefixes=CHECK-GPU-BUILTIN,CHECK-GPU-CUSTOM --input-file %t.opencl.out

// CHECK-GPU-BUILTIN: gpu_selector(){{.*}}GPU : OpenCL
// CHECK-GPU-CUSTOM: custom_selector(gpu){{.*}}GPU : OpenCL

//==-- sycl-ls-gpu-opencl.cpp - SYCL test for discovered/selected devices -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
