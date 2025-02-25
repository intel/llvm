// REQUIRES: opencl-aot, cpu
// REQUIRES: build-and-run-mode

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 %S/Inputs/common.cpp -o %t.out \
// RUN:          -fsycl-dead-args-optimization
// RUN: %{run} %t.out

// This test checks correctness of SYCL2020 non-native specialization constants
// on CPU device
