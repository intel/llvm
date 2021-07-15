// REQUIRES: gpu, rocm, sycl-ls

// RUN: env SYCL_DEVICE_FILTER=rocm sycl-ls --verbose >%t.rocm.out
// RUN: FileCheck %s --check-prefixes=CHECK-BUILTIN-GPU-ROCM,CHECK-CUSTOM-GPU-ROCM --input-file %t.rocm.out

// CHECK-BUILTIN-GPU-ROCM: gpu_selector(){{.*}}GPU :{{.*}}ROCM
// CHECK-CUSTOM-GPU-ROCM: custom_selector(gpu){{.*}}GPU :{{.*}}ROCM

//==---- sycl-ls-gpu-rocm.cpp - SYCL test for discovered/selected devices --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
