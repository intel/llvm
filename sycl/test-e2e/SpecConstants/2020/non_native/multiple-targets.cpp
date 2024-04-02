// REQUIRES: opencl-aot, cpu
// UNSUPPORTED: cuda

// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_x86_64 %S/Inputs/common.cpp -o %t_spv_cpu.out
// RUN: %{run} %t_spv_cpu.out
// Ensure that image ordering does not impact the execution
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64,spir64 %S/Inputs/common.cpp -o %t_cpu_spv.out
// RUN: %{run} %t_cpu_spv.out

// This test checks correctness of SYCL2020 specialization constants' handling
// whenever targets with both native and non-native support are present for the
// fat binary.
