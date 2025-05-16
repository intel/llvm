//==-- multiple-devices.cpp - Appropriate AOT-compiled image selection -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: opencl-aot, ocloc, any-device-is-cpu, any-device-is-gpu, target-spir, opencl-cpu-rt

// Produce a fat object for all targets (generic SPIR-V, CPU, GPU)
// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_x86_64,spir64_gen %S/Inputs/aot.cpp -c -o %t.o

// CPU, GPU
// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %t.o -o %t_cpu_gpu.out
// RUN: %{run} %t_cpu_gpu.out

// No AOT-compiled image for CPU
// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %t.o -o %t_spv_gpu.out
// Check that execution on AOT-compatible devices is unaffected
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t_spv_gpu.out

// No AOT-compiled image for GPU
// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=spir64,spir64_x86_64 %t.o -o %t_spv_cpu.out
// Check that execution on AOT-compatible devices is unaffected
// RUN: env ONEAPI_DEVICE_SELECTOR="*:cpu" %{run-unfiltered-devices} %t_spv_cpu.out

// No AOT-compiled image for FPGA
// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=spir64,spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %t.o -o %t_spv_cpu_gpu.out
// Check that execution on AOT-compatible devices is unaffected
// RUN: %{run} %t_spv_cpu_gpu.out
