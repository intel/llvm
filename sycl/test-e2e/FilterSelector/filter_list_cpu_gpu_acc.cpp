//==------------------ filter_list_cpu_gpu_acc.cpp ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

// REQUIRES: cpu, gpu, accelerator, host

// RUN: %clangxx -fsycl %S/Inputs/filter_list_queries.cpp -o %t.out

// RUN: env ONEAPI_DEVICE_SELECTOR="*:acc" %t.out | FileCheck %s --check-prefixes=CHECK-ACC-ONLY
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %t.out | FileCheck %s --check-prefixes=CHECK-GPU-ONLY
// RUN: env ONEAPI_DEVICE_SELECTOR="*:cpu" %t.out | FileCheck %s --check-prefixes=CHECK-CPU-ONLY
// RUN: env ONEAPI_DEVICE_SELECTOR="host:*" %t.out | FileCheck %s --check-prefixes=CHECK-HOST-ONLY
//
// RUN: env ONEAPI_DEVICE_SELECTOR="*:acc,gpu" %t.out | FileCheck %s --check-prefixes=CHECK-ACC-GPU
// RUN: env ONEAPI_DEVICE_SELECTOR="*:acc,cpu" %t.out | FileCheck %s --check-prefixes=CHECK-ACC-CPU
// RUN: env ONEAPI_DEVICE_SELECTOR="*:cpu,host:*" %t.out | FileCheck %s --check-prefixes=CHECK-CPU-HOST
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu,host:*" %t.out | FileCheck %s --check-prefixes=CHECK-GPU-HOST
//
// RUN: env ONEAPI_DEVICE_SELECTOR="*:cpu,acc,gpu" %t.out | FileCheck %s --check-prefixes=CHECK-ACC-GPU-CPU
//
// CHECK-ACC-ONLY: Device: acc
// CHECK-ACC-ONLY-NOT: Device: cpu
// CHECK-ACC-ONLY-NOT: Device: gpu
// CHECK-ACC-ONLY-NOT: Device: host
//
// CHECK-GPU-ONLY-NOT: Device: acc
// CHECK-GPU-ONLY: Device: gpu
// CHECK-GPU-ONLY-NOT: Device: cpu
// CHECK-GPU-ONLY-NOT: Device: host
//
// CHECK-CPU-ONLY-NOT: Device: acc
// CHECK-CPU-ONLY: Device: cpu
// CHECK-CPU-ONLY-NOT: Device: gpu
// CHECK-CPU-ONLY-NOT: Device: host
//
// CHECK-HOST-ONLY-NOT: Device: acc
// CHECK-HOST-ONLY-NOT: Device: gpu
// CHECK-HOST-ONLY-NOT: Device: cpu
// CHECK-HOST-ONLY: Device: host
//
// CHECK-ACC-GPU: Device: acc
// CHECK-ACC-GPU-NEXT: Device: gpu
// CHECK-ACC-GPU-NOT: Device: cpu
// CHECK-ACC-GPU-NOT: Device: host
//
// CHECK-CPU-HOST-NOT: Device: acc
// CHECK-CPU-HOST-NOT: Device: gpu
// CHECK-CPU-HOST: Device: cpu
// CHECK-CPU-HOST: Device: host
//
// CHECK-ACC-CPU: Device: acc
// CHECK-ACC-CPU-NEXT: Device: cpu
// CHECK-ACC-CPU-NOT: Device: gpu
// CHECK-ACC-CPU-NOT: Device: host
//
// CHECK-GPU-HOST-NOT: Device: acc
// CHECK-GPU-HOST: Device: gpu
// CHECK-GPU-HOST-NOT: Device: cpu
// CHECK-GPU-HOST: Device: host
//
// CHECK-ACC-CPU-HOST: Device: acc
// CHECK-ACC-CPU-HOST: Device: cpu
// CHECK-ACC-CPU-HOST-NOT: Device: gpu
// CHECK-ACC-CPU-HOST: Device: host
//
// CHECK-ACC-GPU-CPU: Device: acc
// CHECK-ACC-GPU-CPU-DAG: Device: gpu
// CHECK-ACC-GPU-CPU-DAG: Device: cpu
// CHECK-ACC-GPU-CPU-NOT: Device: host
