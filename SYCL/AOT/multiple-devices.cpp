//==-- multiple-devices.cpp - Appropriate AOT-compiled image selection -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: opencl-aot, ocloc, aoc, cpu, gpu, accelerator, llvm-link, llvm-spirv
// UNSUPPORTED: cuda
// CUDA is not compatible with SPIR.

// The test is failing with GPU RT 30.0.100.9667
// XFAIL: windows

// 1-command compilation case
// Targeting CPU, GPU, FPGA
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice,spir64_gen-unknown-unknown-sycldevice,spir64_fpga-unknown-unknown-sycldevice -Xsycl-target-backend=spir64_gen-unknown-unknown-sycldevice "-device *" %S/Inputs/aot.cpp -o %t_all.out
// RUN: %HOST_RUN_PLACEHOLDER %t_all.out
// RUN: %CPU_RUN_PLACEHOLDER %t_all.out
// RUN: %GPU_RUN_PLACEHOLDER %t_all.out
// RUN: %ACC_RUN_PLACEHOLDER %t_all.out

// FIXME: Change the behavior when proper automation for assert support is
// introduced. For the time being, AOT flow can't detect if specific extension
// is available for this or that device. The automation to be introduced is to
// query native binary generator on specific features.
//
// Produce object file, spirv, device images to combine these differently
// at link-time, thus testing various AOT-compiled images configurations
// RUN: %clangxx -fsycl %S/Inputs/aot.cpp -c -o %t.o
// RUN: %clangxx -fsycl -fsycl-link-targets=spir64-unknown-unknown-sycldevice %t.o -o %t.spv
// AOT-compile device binary images
// Neither of AOT tools can compile several files, hence, here is this
// workaround
// RUN: %llvm_spirv -r %sycl_libs_dir/libsycl-fallback-cassert.spv -o=%T/fallback-cassert.bc
// RUN: %llvm_spirv -r %t.spv -o=%t.bc
// RUN: %llvm_link %t.bc %T/fallback-cassert.bc -o=%t2.bc
// RUN: %llvm_spirv %t2.bc -o=%t.spv
// RUN: opencl-aot %t.spv -o=%t_cpu.ir --device=cpu
// RUN: ocloc -file %t.spv -spirv_input -output %t_gen.out -output_no_suffix -device cfl
// RUN: aoc %t.spv -o %t_fpga.aocx -sycl -dep-files=%t.d

// CPU, GPU
// RUN: %clangxx -fsycl -fsycl-add-targets=spir64_x86_64:%t_cpu.ir,spir64_gen:%t_gen.out %t.o -o %t_cpu_gpu.out
// RUN: %HOST_RUN_PLACEHOLDER %t_cpu_gpu.out
// RUN: %CPU_RUN_PLACEHOLDER %t_cpu_gpu.out
// RUN: %GPU_RUN_PLACEHOLDER %t_cpu_gpu.out

// CPU, FPGA
// RUN: %clangxx -fsycl -fsycl-add-targets=spir64_x86_64:%t_cpu.ir,spir64_fpga:%t_fpga.aocx %t.o -o %t_cpu_fpga.out
// RUN: %HOST_RUN_PLACEHOLDER %t_cpu_fpga.out
// RUN: %CPU_RUN_PLACEHOLDER %t_cpu_fpga.out
// RUN: %ACC_RUN_PLACEHOLDER %t_cpu_fpga.out

// GPU, FPGA
// RUN: %clangxx -fsycl -fsycl-add-targets=spir64_gen:%t_gen.out,spir64_fpga:%t_fpga.aocx %t.o -o %t_gpu_fpga.out
// RUN: %HOST_RUN_PLACEHOLDER %t_gpu_fpga.out
// RUN: %GPU_RUN_PLACEHOLDER %t_gpu_fpga.out
// RUN: %ACC_RUN_PLACEHOLDER %t_gpu_fpga.out

// No AOT-compiled image for CPU
// RUN: %clangxx -fsycl -fsycl-add-targets=spir64:%t.spv,spir64_gen:%t_gen.out,spir64_fpga:%t_fpga.aocx %t.o -o %t_spv_gpu_fpga.out
// RUN: %CPU_RUN_PLACEHOLDER %t_spv_gpu_fpga.out
// Check that execution on AOT-compatible devices is unaffected
// RUN: %GPU_RUN_PLACEHOLDER %t_spv_gpu_fpga.out
// RUN: %ACC_RUN_PLACEHOLDER %t_spv_gpu_fpga.out

// No AOT-compiled image for GPU
// RUN: %clangxx -fsycl -fsycl-add-targets=spir64:%t.spv,spir64_x86_64:%t_cpu.ir,spir64_fpga:%t_fpga.aocx %t.o -o %t_spv_cpu_fpga.out
// RUN: %GPU_RUN_PLACEHOLDER %t_spv_cpu_fpga.out
// Check that execution on AOT-compatible devices is unaffected
// RUN: %CPU_RUN_PLACEHOLDER %t_spv_cpu_fpga.out
// RUN: %ACC_RUN_PLACEHOLDER %t_spv_cpu_fpga.out

// No AOT-compiled image for FPGA
// RUN: %clangxx -fsycl -fsycl-add-targets=spir64:%t.spv,spir64_x86_64:%t_cpu.ir,spir64_gen:%t_gen.out %t.o -o %t_spv_cpu_gpu.out
// RUN: %ACC_RUN_PLACEHOLDER %t_spv_cpu_gpu.out
// Check that execution on AOT-compatible devices is unaffected
// RUN: %CPU_RUN_PLACEHOLDER %t_spv_cpu_gpu.out
// RUN: %GPU_RUN_PLACEHOLDER %t_spv_cpu_gpu.out
