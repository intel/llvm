// REQUIRES: opencl-aot, cpu

// CPU AOT targets host isa, so we compile on the run system instead.
// Test with `--offload-new-driver`
// RUN: %{run-aux} %clangxx -fsycl -fsycl-device-code-split=per_source -fsycl-targets=spir64_x86_64 -I %S/Inputs -o %t.out %S/split-per-source-main.cpp %S/Inputs/split-per-source-second-file.cpp \
// RUN: -fsycl-dead-args-optimization --offload-new-driver
// RUN: %{run} %t.out

// Test -O0 with `--offload-new-driver`
// RUN: %{run-aux} %clangxx %O0 -fsycl -fsycl-targets=spir64-x86_64 %S/Inputs/aot.cpp
// RUN: %{run} %t.out
