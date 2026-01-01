//==--------------- aot-multiple-device.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// End-to-end test for verifying that when compiling a fat binary for multiple
// device targets, all targets from the fat binary are processed during linking,
// regardless of which specific targets are specified in the link command
// line.

// This test is copied from AOT/multiple_devices.cpp
// and modified to test with the New Offloading Model.

// REQUIRES: opencl-aot, ocloc, any-device-is-cpu, any-device-is-gpu, target-spir, opencl-cpu-rt

// Produce a fat object for all targets (generic SPIR-V, CPU, GPU).
// It is required to specify the device name when compiling AOT for GPU.
// RUN: %{run-aux} %clangxx --offload-new-driver -fsycl -fsycl-targets=spir64,spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %S/Inputs/aot.cpp -c -o %t.o

// RUN: %{run-aux} %clangxx --offload-new-driver -fsycl -fsycl-targets=spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts -v %t.o -o %t_cpu_gpu.out 2>&1 | FileCheck %s --check-prefix=CHECK-DEVICE
// RUN: %{run} %t_cpu_gpu.out

// Check that all targets compiled in the fat binary are selected during
// linking (including CPU even though it is not specified during linking).
// RUN: %{run-aux} %clangxx --offload-new-driver -fsycl -fsycl-targets=spir64,spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts -v %t.o -o %t_spv_gpu.out 2>&1 | FileCheck %s --check-prefix=CHECK-DEVICE
// Check that execution on AOT-compatible devices is unaffected
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t_spv_gpu.out

// Check that all targets compiled in the fat binary are selected during
// linking (including GPU even though it is not specified during linking).
// RUN: %{run-aux} %clangxx --offload-new-driver -fsycl -fsycl-targets=spir64,spir64_x86_64 -v %t.o -o %t_spv_cpu.out 2>&1 | FileCheck %s --check-prefix=CHECK-DEVICE
// Check that execution on AOT-compatible devices is unaffected.
// RUN: env ONEAPI_DEVICE_SELECTOR="*:cpu" %{run-unfiltered-devices} %t_spv_cpu.out

// CHECK-DEVICE-DAG: spirv-to-ir-wrapper{{.*}}spir64_gen-unknown-unknown{{.*}}
// CHECK-DEVICE-DAG: spirv-to-ir-wrapper{{.*}}spir64_x86_64-unknown-unknown{{.*}}
// CHECK-DEVICE-DAG: spirv-to-ir-wrapper{{.*}}spir64-unknown-unknown{{.*}}
