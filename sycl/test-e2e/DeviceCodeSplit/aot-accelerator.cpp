// REQUIRES: opencl-aot, accelerator

// RUN: %clangxx -fsycl -fsycl-device-code-split=per_source -fsycl-targets=spir64_fpga -I %S/Inputs -o %t.out %S/split-per-source-main.cpp %S/Inputs/split-per-source-second-file.cpp \
// RUN: -fsycl-dead-args-optimization
// RUN: %ACC_RUN_PLACEHOLDER %t.out
