// REQUIRES: aoc, accelerator

// RUN: %clangxx -fsycl -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice %S/Inputs/common.cpp -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test checks correctness of SYCL2020 non-native specialization constants
// on accelerator device
