// REQUIRES: aoc, accelerator

// RUN: %clangxx -fsycl -fsycl-device-code-split=per_source -fsycl-targets=spir64_fpga-unknown-unknown-sycldevice -I %S/Inputs -o %t.out %S/split-per-source-main.cpp %S/Inputs/split-per-source-second-file.cpp
// RUN: %ACC_RUN_PLACEHOLDER %t.out
