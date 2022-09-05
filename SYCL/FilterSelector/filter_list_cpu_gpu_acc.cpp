//==------------------ filter_list_cpu_gpu_acc.cpp ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

// REQUIRES: cpu, gpu, accelerator

// RUN: %clangxx -fsycl %S/Inputs/filter_list_queries.cpp -o %t.out

// RUN: env SYCL_DEVICE_FILTER=acc %t.out | FileCheck %s --check-prefixes=CHECK-ACC-ONLY
// RUN: env SYCL_DEVICE_FILTER=gpu %t.out | FileCheck %s --check-prefixes=CHECK-GPU-ONLY
// RUN: env SYCL_DEVICE_FILTER=cpu %t.out | FileCheck %s --check-prefixes=CHECK-CPU-ONLY
//
// RUN: env SYCL_DEVICE_FILTER=acc,gpu %t.out | FileCheck %s --check-prefixes=CHECK-ACC-GPU
// RUN: env SYCL_DEVICE_FILTER=acc,cpu %t.out | FileCheck %s --check-prefixes=CHECK-ACC-CPU
//
// RUN: env SYCL_DEVICE_FILTER=cpu,acc,gpu %t.out | FileCheck %s --check-prefixes=CHECK-ACC-GPU-CPU
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
