// REQUIRES: gpu, cuda

// RUN: %{run-unfiltered-devices} env ONEAPI_DEVICE_SELECTOR="cuda:*" sycl-ls --verbose | \
// RUN: FileCheck %s --check-prefixes=CHECK-BUILTIN-GPU-CUDA,CHECK-CUSTOM-GPU-CUDA

// CHECK-BUILTIN-GPU-CUDA: gpu_selector(){{.*}}gpu, {{.*}}CUDA
// CHECK-CUSTOM-GPU-CUDA: custom_selector(gpu){{.*}}gpu, {{.*}}CUDA

//==---- sycl-ls-gpu-cuda.cpp - SYCL test for discovered/selected devices --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
