// REQUIRES: gpu, level-zero

// TODO: Remove unsetting SYCL_DEVICE_FILTER when feature is dropped
// RUN: env --unset=SYCL_DEVICE_FILTER --unset=ONEAPI_DEVICE_SELECTOR sycl-ls --verbose >%t.default.out
// RUN: FileCheck %s --check-prefixes=CHECK-GPU-BUILTIN,CHECK-GPU-CUSTOM --input-file %t.default.out

// CHECK-GPU-BUILTIN: gpu_selector(){{.*}}gpu, {{.*}}Level-Zero
// CHECK-GPU-CUSTOM: custom_selector(gpu){{.*}}gpu, {{.*}}Level-Zero

//==------------------ sycl-ls-gpu-default-level-zero.cpp ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test checks that, if available, a Level-Zero GPU will be selected by
// the default GPU selector.
