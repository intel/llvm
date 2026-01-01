//==-- multiple-devices.cpp - Appropriate AOT-compiled image selection -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// End-to-end test for verifying that the targets specified at link command
// line are used to select images from the fat binary built for multiple
// targets, and that compilation occurs for those targets.

// UNSUPPORTED-INTENDED:  device names are expected to be specified when
// compiling AOT fat binaries. This test should be removed, when old offload
// model support is removed.
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20988
//
// Explanation: Old offloading model was accepting compilation of a fat binary
// for AOT target without specifying specific device. In that case specific
// device to AOT-compile for was specified at link command line and compilation
// was happening only for targets specified in link command line, ignoring
// targets specified in compile command line, which doesn't look correct. In the
// new offloading model, the AOT images for all targets specified in compile
// command line are honored and AOT-compiled at link stage. To make it correct,
// if AOT target is specified at compile time, specific device should also be
// provided at compilation command line.
//
// A modified test can be found at NewOffloadDriver/aot-multiple-device.cpp

// REQUIRES: opencl-aot, ocloc, any-device-is-cpu, any-device-is-gpu, target-spir, opencl-cpu-rt

// Produce a fat object for all targets (generic SPIR-V, CPU, GPU)
// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_x86_64,spir64_gen %S/Inputs/aot.cpp -c -o %t.o

// Verify that AOT compilation occurs for the device targets (GPU, CPU)
// specified in the link command line. Note that generic SPIR-V compilation
// is enabled by default even when only GPU or CPU targets are specified.
// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts -v %t.o -o %t_cpu_gpu.out 2>&1 | FileCheck %s --check-prefix=CHECK-AOT-CPU --check-prefix=CHECK-AOT-GPU --check-prefix=CHECK-GENERIC
// RUN: %{run} %t_cpu_gpu.out

// Verify that AOT compilation occurs for the device targets (GPU) specified in
// the link command line.
// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts -v %t.o -o %t_spv_gpu.out 2>&1 | FileCheck %s --check-prefix=CHECK-GENERIC --check-prefix=CHECK-AOT-GPU --check-prefix=CHECK-NO-AOT-CPU
// Check that execution on AOT-compatible devices is unaffected
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t_spv_gpu.out

// Verify that AOT compilation occurs for the device targets (CPU) specified in
// the link command line.
// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=spir64,spir64_x86_64 -v %t.o -o %t_spv_cpu.out 2>&1 | FileCheck %s --check-prefix=CHECK-GENERIC --check-prefix=CHECK-AOT-CPU --check-prefix=CHECK-NO-AOT-GPU
// Check that execution on AOT-compatible devices is unaffected
// RUN: env ONEAPI_DEVICE_SELECTOR="*:cpu" %{run-unfiltered-devices} %t_spv_cpu.out

// CHECK-GENERIC-DAG: clang-offload-wrapper{{.*}} -target=spir64{{[^_]}}
// CHECK-AOT-CPU-DAG: clang-offload-wrapper{{.*}} -target=spir64_x86_64
// CHECK-AOT-GPU-DAG: clang-offload-wrapper{{.*}} -target=spir64_gen
// CHECK-NO-AOT-CPU-NOT: clang-offload-wrapper{{.*}} -target=spir64_x86_64
// CHECK-NO-AOT-GPU-NOT: clang-offload-wrapper{{.*}} -target=spir64_gen
