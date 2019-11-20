// REQUIRES: aoc

// RUN: %clangxx -fsycl -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice -I %S/Inputs -o %t.out %S/split-per-source-main.cpp %S/Inputs/split-per-source-second-file.cpp
// RUN: %ACC_RUN_PLACEHOLDER %t.out
