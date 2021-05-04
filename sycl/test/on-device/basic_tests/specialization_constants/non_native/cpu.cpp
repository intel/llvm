// REQUIRES: opencl-aot, cpu

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice %S/Inputs/common.cpp -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

// This test checks correctness of SYCL2020 non-native specialization constants
// on CPU device
