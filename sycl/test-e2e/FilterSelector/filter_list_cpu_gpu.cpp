//==------------------ filter_list_cpu_gpu.cpp ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

// REQUIRES: any-device-is-cpu, any-device-is-gpu

// RUN: %clangxx -fsycl %S/Inputs/filter_list_queries.cpp -o %t.out

// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out | FileCheck %s --check-prefixes=CHECK-GPU-ONLY
// RUN: env ONEAPI_DEVICE_SELECTOR="*:cpu" %{run-unfiltered-devices} %t.out | FileCheck %s --check-prefixes=CHECK-CPU-ONLY
// RUN: env ONEAPI_DEVICE_SELECTOR="*:cpu,gpu" %{run-unfiltered-devices} %t.out | FileCheck %s --check-prefixes=CHECK-GPU-CPU
//
// CHECK-GPU-ONLY: Device: gpu
// CHECK-GPU-ONLY-NOT: Device: cpu
//
// CHECK-CPU-ONLY: Device: cpu
// CHECK-CPU-ONLY-NOT: Device: gpu
//
// CHECK-GPU-CPU: Device: gpu
// CHECK-GPU-CPU-DAG: Device: cpu
