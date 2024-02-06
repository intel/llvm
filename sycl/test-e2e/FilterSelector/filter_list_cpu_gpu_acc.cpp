//==------------------ filter_list_cpu_gpu_acc.cpp ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

// REQUIRES: cpu, gpu, accelerator

// RUN: %clangxx -fsycl %S/Inputs/filter_list_queries.cpp -o %t.out

// RUN: env ONEAPI_DEVICE_SELECTOR="*:acc" %{run-unfiltered-devices} %t.out | FileCheck %s --check-prefixes=CHECK-ACC-ONLY
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out | FileCheck %s --check-prefixes=CHECK-GPU-ONLY
// RUN: env ONEAPI_DEVICE_SELECTOR="*:cpu" %{run-unfiltered-devices} %t.out | FileCheck %s --check-prefixes=CHECK-CPU-ONLY
//
// RUN: env ONEAPI_DEVICE_SELECTOR="*:acc,gpu" %{run-unfiltered-devices} %t.out | FileCheck %s --check-prefixes=CHECK-ACC-GPU
// RUN: env ONEAPI_DEVICE_SELECTOR="*:acc,cpu" %{run-unfiltered-devices} %t.out | FileCheck %s --check-prefixes=CHECK-ACC-CPU
//
// RUN: env ONEAPI_DEVICE_SELECTOR="*:cpu,acc,gpu" %{run-unfiltered-devices} %t.out | FileCheck %s --check-prefixes=CHECK-ACC-GPU-CPU
//
// CHECK-ACC-ONLY: Device: acc
// CHECK-ACC-ONLY-NOT: Device: cpu
// CHECK-ACC-ONLY-NOT: Device: gpu
//
// CHECK-GPU-ONLY-NOT: Device: acc
// CHECK-GPU-ONLY: Device: gpu
// CHECK-GPU-ONLY-NOT: Device: cpu
//
// CHECK-CPU-ONLY-NOT: Device: acc
// CHECK-CPU-ONLY: Device: cpu
// CHECK-CPU-ONLY-NOT: Device: gpu
//
// CHECK-ACC-GPU: Device: acc
// CHECK-ACC-GPU-NEXT: Device: gpu
// CHECK-ACC-GPU-NOT: Device: cpu
//
// CHECK-ACC-CPU: Device: acc
// CHECK-ACC-CPU-NEXT: Device: cpu
// CHECK-ACC-CPU-NOT: Device: gpu
//
// CHECK-ACC-GPU-CPU: Device: acc
// CHECK-ACC-GPU-CPU-DAG: Device: gpu
// CHECK-ACC-GPU-CPU-DAG: Device: cpu
