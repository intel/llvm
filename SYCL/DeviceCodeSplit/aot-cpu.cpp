// REQUIRES: opencl-aot, cpu

// RUN: %clangxx -fsycl -fsycl-device-code-split=per_source -fsycl-targets=spir64_x86_64 -I %S/Inputs -o %t.out %S/split-per-source-main.cpp %S/Inputs/split-per-source-second-file.cpp
// RUN: %CPU_RUN_PLACEHOLDER %t.out
