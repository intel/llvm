//==-- multiple-devices.cpp - Appropriate AOT-compiled image selection -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: opencl-aot, ocloc, cpu, gpu, accelerator
// UNSUPPORTED: cuda
// CUDA is not compatible with SPIR.

// Produce a fat object for all targets (generic SPIR-V, CPU, GPU, FPGA)
// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_x86_64,spir64_gen,spir64_fpga %S/Inputs/aot.cpp -c -o %t.o

// CPU, GPU, FPGA
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64,spir64_gen,spir64_fpga -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %t.o -o %t_all_aot.out
// RUN: %HOST_RUN_PLACEHOLDER %t_all_aot.out
// RUN: %CPU_RUN_PLACEHOLDER %t_all_aot.out
// RUN: %GPU_RUN_PLACEHOLDER %t_all_aot.out
// RUN: %ACC_RUN_PLACEHOLDER %t_all_aot.out

// CPU, GPU
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %t.o -o %t_cpu_gpu.out
// RUN: %HOST_RUN_PLACEHOLDER %t_cpu_gpu.out
// RUN: %CPU_RUN_PLACEHOLDER %t_cpu_gpu.out
// RUN: %GPU_RUN_PLACEHOLDER %t_cpu_gpu.out

// CPU, FPGA
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64,spir64_fpga %t.o -o %t_cpu_fpga.out
// RUN: %HOST_RUN_PLACEHOLDER %t_cpu_fpga.out
// RUN: %CPU_RUN_PLACEHOLDER %t_cpu_fpga.out
// RUN: %ACC_RUN_PLACEHOLDER %t_cpu_fpga.out

// GPU, FPGA
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen,spir64_fpga -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %t.o -o %t_gpu_fpga.out
// RUN: %HOST_RUN_PLACEHOLDER %t_gpu_fpga.out
// RUN: %GPU_RUN_PLACEHOLDER %t_gpu_fpga.out
// RUN: %ACC_RUN_PLACEHOLDER %t_gpu_fpga.out

// No AOT-compiled image for CPU
// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_gen,spir64_fpga -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %t.o -o %t_spv_gpu_fpga.out
// RUN: %CPU_RUN_PLACEHOLDER %t_spv_gpu_fpga.out
// Check that execution on AOT-compatible devices is unaffected
// RUN: %GPU_RUN_PLACEHOLDER %t_spv_gpu_fpga.out
// RUN: %ACC_RUN_PLACEHOLDER %t_spv_gpu_fpga.out

// No AOT-compiled image for GPU
// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_x86_64,spir64_fpga %t.o -o %t_spv_cpu_fpga.out
// RUN: %GPU_RUN_PLACEHOLDER %t_spv_cpu_fpga.out
// Check that execution on AOT-compatible devices is unaffected
// RUN: %CPU_RUN_PLACEHOLDER %t_spv_cpu_fpga.out
// RUN: %ACC_RUN_PLACEHOLDER %t_spv_cpu_fpga.out

// No AOT-compiled image for FPGA
// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %t.o -o %t_spv_cpu_gpu.out
// RUN: %ACC_RUN_PLACEHOLDER %t_spv_cpu_gpu.out
// Check that execution on AOT-compatible devices is unaffected
// RUN: %CPU_RUN_PLACEHOLDER %t_spv_cpu_gpu.out
// RUN: %GPU_RUN_PLACEHOLDER %t_spv_cpu_gpu.out
