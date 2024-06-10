// REQUIRES: gpu

// TODO: Remove unsetting SYCL_DEVICE_FILTER when feature is dropped
// RUN: env --unset=SYCL_DEVICE_FILTER  --unset=ONEAPI_DEVICE_SELECTOR sycl-ls --verbose >%t.default.out
// RUN: FileCheck %s --check-prefixes=CHECK-GPU-BUILTIN,CHECK-GPU-CUSTOM --input-file %t.default.out

// CHECK-GPU-BUILTIN: gpu_selector(){{.*}}gpu, {{.*}}{{Level-Zero|CUDA|OpenCL}}
// clang-format off
// CHECK-GPU-CUSTOM: custom_selector(gpu){{.*}}gpu, {{.*}}{{Level-Zero|CUDA|OpenCL}}
// clang-format on

//==--------------------- sycl-ls-gpu-default-any.cpp ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test checks that a valid GPU is returned by sycl-ls by default if one
// is present.
// UNSUPPORTED: hip
// Temporarily disable on L0 due to fails in CI
// UNSUPPORTED: level_zero
