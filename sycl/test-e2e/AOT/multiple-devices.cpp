//==-- multiple-devices.cpp - Appropriate AOT-compiled image selection -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// End-to-End test for testing appropriate AOT-compiled device image got selected 
// correctly from a fat binary built for multiple device targets

// UNSUPPORTED: target={{.*}}

// This test should not be supported because device names should be specified 
// when compiling AOT fat binaries for GPU targets.
//
// Explanation: In the old offloading model, when compiling a fat binary for 
// multiple device targets A, B, C and linking only with targets A and B, 
// the AOT images for only targets A and B are selected from the fat binary. 
// However, in the new offloading model, the AOT images for all targets A, B, C 
// will be selected during linking even though only A and B are specified. 
// Therefore, when compiling AOT fat binaries for GPU, the device name must be 
// specified, otherwise linking will fail due to missing device names.
//
// A modified test can be found at NewOffloadDriver/aot-multiple-device.cpp

// REQUIRES: opencl-aot, ocloc, any-device-is-cpu, any-device-is-gpu, target-spir, opencl-cpu-rt

// Produce a fat object for all targets (generic SPIR-V, CPU, GPU)
// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_x86_64,spir64_gen %S/Inputs/aot.cpp -c -o %t.o

// AOT image selection with CPU and GPU targets available
// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %t.o -o %t_cpu_gpu.out
// RUN: %{run} %t_cpu_gpu.out

// AOT image selection with generic SPIR-V and GPU target available
// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %t.o -o %t_spv_gpu.out
// Check that execution on AOT-compatible devices is unaffected
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t_spv_gpu.out

// AOT image selection with generic SPIR-V and CPU target available
// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=spir64,spir64_x86_64 %t.o -o %t_spv_cpu.out
// Check that execution on AOT-compatible devices is unaffected
// RUN: env ONEAPI_DEVICE_SELECTOR="*:cpu" %{run-unfiltered-devices} %t_spv_cpu.out

// AOT image selection with all targets available (SPIR-V, CPU, GPU)
// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=spir64,spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %t.o -o %t_spv_cpu_gpu.out
// Check that execution on AOT-compatible devices is unaffected
// RUN: %{run} %t_spv_cpu_gpu.out
