// REQUIRES: gpu, hip, sycl-ls

// RUN: %{run-unfiltered-devices} env ONEAPI_DEVICE_SELECTOR="hip:*" sycl-ls --verbose | \
// RUN: FileCheck %s --check-prefixes=CHECK-BUILTIN-GPU-HIP,CHECK-CUSTOM-GPU-HIP

// CHECK-BUILTIN-GPU-HIP: gpu_selector(){{.*}}gpu, {{.*}}HIP
// CHECK-CUSTOM-GPU-HIP: custom_selector(gpu){{.*}}gpu, {{.*}}HIP

//==---- sycl-ls-gpu-hip.cpp - SYCL test for discovered/selected devices --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
