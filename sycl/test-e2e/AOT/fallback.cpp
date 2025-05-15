//==-- fallback.cpp - Fallback to JIT if there is no appropriate AOT image--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: ocloc, any-device-is-gpu, target-spir
// Remove support for platform used as compile target since AOT image should be
// not applicable.
// UNSUPPORTED: arch-intel_gpu_tgl

// AOT-compiled image for absent gen platform, run on GPU
// RUN: %clangxx -fsycl -fsycl-targets=spir64,intel_gpu_tgl %S/Inputs/aot.cpp -o %t_spv_gpu.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" SYCL_UR_TRACE=2 %{run-unfiltered-devices} %t_spv_gpu.out | FileCheck %s

// CHECK: ---> urProgramCreateWithIL
// CHECK-NOT: ---> urProgramCreateWithBinary
