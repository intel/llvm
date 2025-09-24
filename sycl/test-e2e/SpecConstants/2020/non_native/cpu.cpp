// REQUIRES: opencl-aot, cpu

// CPU AOT targets host isa, so we compile on the run system instead.
// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64 %S/Inputs/common.cpp -o %t.out \
// RUN:          -fsycl-dead-args-optimization
// RUN: %{run} %t.out

// UNSUPPORTED: target-native_cpu
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20142

// This test checks correctness of SYCL2020 non-native specialization constants
// on CPU device
