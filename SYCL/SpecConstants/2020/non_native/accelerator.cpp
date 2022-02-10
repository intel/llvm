// REQUIRES: opencl-aot, accelerator

// RUN: %clangxx -fsycl -fsycl-targets=spir64_fpga %S/Inputs/common.cpp -o %t.out \
// RUN:          -fsycl-dead-args-optimization
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test checks correctness of SYCL2020 non-native specialization constants
// on accelerator device
