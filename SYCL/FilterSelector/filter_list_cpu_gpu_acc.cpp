//==------------------ filter_list_cpu_gpu_acc.cpp ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

// REQUIRES: opencl, cpu, gpu, accelerator

// RUN: %clangxx -fsycl %S/Inputs/filter_list_queries.cpp -o %t.out

// RUN: env SYCL_DEVICE_FILTER="*" %t.out | FileCheck %s --check-prefixes=CHECK-CPU,CHECK-GPU,CHECK-ACC,CHECK-HOST
// RUN: env SYCL_DEVICE_FILTER=opencl,host %t.out | FileCheck %s --check-prefixes=CHECK-CPU,CHECK-GPU,CHECK-ACC,CHECK-HOST
// RUN: env SYCL_DEVICE_FILTER=cpu,host %t.out | FileCheck %s --check-prefixes=CHECK-CPU,CHECK-GPU-NOT,CHECK-ACC-NOT,CHECK-HOST
// RUN: env SYCL_DEVICE_FILTER=acc,host %t.out | FileCheck %s --check-prefixes=CHECK-CPU-NOT,CHECK-GPU-NOT,CHECK-ACC,CHECK-HOST
// RUN: env SYCL_DEVICE_FILTER=host %t.out | FileCheck %s --check-prefixes=CHECK-CPU-NOT,CHECK-GPU-NOT,CHECK-ACC-NOT,CHECK-HOST
// RUN: env SYCL_DEVICE_FILTER=gpu,host %t.out | FileCheck %s --check-prefixes=CHECK-CPU-NOT,CHECK-GPU,CHECK-ACC-NOT,CHECK-HOST
// RUN: env SYCL_DEVICE_FILTER=cpu,acc,host %t.out | FileCheck %s --check-prefixes=CHECK-CPU,CHECK-GPU-NOT,CHECK-ACC,CHECK-HOST
// RUN: env SYCL_DEVICE_FILTER=cpu:0,acc:1,host %t.out | FileCheck %s --check-prefixes=CHECK-CPU,CHECK-GPU-NOT,CHECK-ACC-NOT,CHECK-HOST
// RUN: env CL_CONFIG_CPU_EMULATE_DEVICES=2 SYCL_DEVICE_FILTER=cpu:0,acc:1,host %t.out | FileCheck %s --check-prefixes=CHECK-CPU,CHECK-GPU-NOT,CHECK-ACC,CHECK-HOST
// RUN: env SYCL_DEVICE_FILTER=cuda:cpu,opencl:gpu:0,level_zero:acc:1,host %t.out | FileCheck %s --check-prefixes=CHECK-CPU-NOT,CHECK-GPU-NOT,CHECK-ACC-NOT,CHECK-HOST
//
// CHECK-ACC: Device: acc
// CHECK-CPU: Device: cpu
// CHECK-GPU: Device: gpu
// CHECK-HOST: Device: host
