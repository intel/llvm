// REQUIRES: opencl-aot, cpu

// CPU AOT targets host isa, so we compile on the run system instead.
// RUN: %{run-aux} %clangxx -fsycl -fsycl-device-code-split=per_source -fsycl-targets=spir64_x86_64 -I %S/Inputs -o %t.out \
// RUN:   %S/split-per-source-main.cpp %S/Inputs/split-per-source-second-file.cpp -fsycl-dead-args-optimization
// RUN: %{run} %t.out

// UNSUPPORTED: target-native_cpu
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20142
