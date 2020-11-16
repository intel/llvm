// REQUIRES: gpu, cuda, opencl

// RUN: sycl-ls --verbose >%t.default.out
// RUN: FileCheck %s --check-prefixes=CHECK-BUILTIN-GPU-OPENCL,CHECK-CUSTOM-GPU-OPENCL --input-file %t.default.out

// RUN: env SYCL_BE=PI_OPENCL sycl-ls --verbose >%t.opencl.out
// RUN: FileCheck %s --check-prefixes=CHECK-BUILTIN-GPU-OPENCL,CHECK-CUSTOM-GPU-OPENCL --input-file %t.opencl.out

// CHECK-BUILTIN-GPU-OPENCL: gpu_selector(){{.*}}GPU : OpenCL
// CHECK-CUSTOM-GPU-OPENCL: custom_selector(gpu){{.*}}GPU : OpenCL

// RUN: env SYCL_BE=PI_CUDA sycl-ls --verbose >%t.cuda.out
// RUN: FileCheck %s --check-prefixes=CHECK-BUILTIN-GPU-CUDA,CHECK-CUSTOM-GPU-CUDA --input-file %t.cuda.out

// CHECK-BUILTIN-GPU-CUDA: gpu_selector(){{.*}}GPU : CUDA
// CHECK-CUSTOM-GPU-CUDA: custom_selector(gpu){{.*}}GPU : CUDA

//==---- sycl-ls-gpu-sycl-be.cpp - SYCL test for discovered/selected devices --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
