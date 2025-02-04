// REQUIRES: gpu, cuda, hip, opencl, sycl-ls

// RUN: %{run-unfiltered-devices} sycl-ls --verbose | \
// RUN: FileCheck %s --check-prefixes=CHECK-BUILTIN-GPU-OPENCL,CHECK-CUSTOM-GPU-OPENCL

// RUN: %{run-unfiltered-devices} env ONEAPI_DEVICE_SELECTOR="opencl:*" sycl-ls --verbose | \
// RUN: FileCheck %s --check-prefixes=CHECK-BUILTIN-GPU-OPENCL,CHECK-CUSTOM-GPU-OPENCL

// CHECK-BUILTIN-GPU-OPENCL: gpu_selector(){{.*}}gpu, {{.*}}OpenCL
// CHECK-CUSTOM-GPU-OPENCL: custom_selector(gpu){{.*}}gpu, {{.*}}OpenCL

// RUN: %{run-unfiltered-devices} env ONEAPI_DEVICE_SELECTOR="cuda:*" sycl-ls --verbose | \
// RUN: FileCheck %s --check-prefixes=CHECK-BUILTIN-GPU-CUDA,CHECK-CUSTOM-GPU-CUDA

// CHECK-BUILTIN-GPU-CUDA: gpu_selector(){{.*}}gpu, {{.*}}CUDA
// CHECK-CUSTOM-GPU-CUDA: custom_selector(gpu){{.*}}gpu, {{.*}}CUDA

// RUN: %{run-unfiltered-devices} env ONEAPI_DEVICE_SELECTOR="hip:*" sycl-ls --verbose | \
// RUN: FileCheck %s --check-prefixes=CHECK-BUILTIN-GPU-HIP,CHECK-CUSTOM-GPU-HIP

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
