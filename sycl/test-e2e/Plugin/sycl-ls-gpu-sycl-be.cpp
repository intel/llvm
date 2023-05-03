// REQUIRES: gpu, cuda, hip, opencl, sycl-ls

// RUN: sycl-ls --verbose >%t.default.out
// RUN: FileCheck %s --check-prefixes=CHECK-BUILTIN-GPU-OPENCL,CHECK-CUSTOM-GPU-OPENCL --input-file %t.default.out

// RUN: env ONEAPI_DEVICE_SELECTOR="opencl:*" sycl-ls --verbose >%t.opencl.out
// RUN: FileCheck %s --check-prefixes=CHECK-BUILTIN-GPU-OPENCL,CHECK-CUSTOM-GPU-OPENCL --input-file %t.opencl.out

// CHECK-BUILTIN-GPU-OPENCL: gpu_selector(){{.*}}gpu, {{.*}}OpenCL
// CHECK-CUSTOM-GPU-OPENCL: custom_selector(gpu){{.*}}gpu, {{.*}}OpenCL

// RUN: env ONEAPI_DEVICE_SELECTOR="cuda:*" sycl-ls --verbose >%t.cuda.out
// RUN: FileCheck %s --check-prefixes=CHECK-BUILTIN-GPU-CUDA,CHECK-CUSTOM-GPU-CUDA --input-file %t.cuda.out

// CHECK-BUILTIN-GPU-CUDA: gpu_selector(){{.*}}gpu, {{.*}}CUDA
// CHECK-CUSTOM-GPU-CUDA: custom_selector(gpu){{.*}}gpu, {{.*}}CUDA

// RUN: env ONEAPI_DEVICE_SELECTOR="hip:*" sycl-ls --verbose >%t.hip.out
// RUN: FileCheck %s --check-prefixes=CHECK-BUILTIN-GPU-HIP,CHECK-CUSTOM-GPU-HIP --input-file %t.hip.out

// CHECK-BUILTIN-GPU-HIP: gpu_selector(){{.*}}gpu, {{.*}}HIP
// CHECK-CUSTOM-GPU-HIP: custom_selector(gpu){{.*}}gpu, {{.*}}HIP

//==---- sycl-ls-gpu-sycl-be.cpp - SYCL test for discovered/selected devices
//--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
