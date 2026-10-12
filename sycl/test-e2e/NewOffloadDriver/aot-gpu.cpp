// REQUIRES: ocloc, gpu, spir-family
// Test with `--offload-new-driver`
//
// RUN: %clangxx -fsycl -fsycl-device-code-split=per_source \
// RUN:   %aot_options -I %S/Inputs -o %t.out \
// RUN:   %S/split-per-source-main.cpp \
// RUN:   %S/Inputs/split-per-source-second-file.cpp \
// RUN:   -fsycl-dead-args-optimization --offload-new-driver
// RUN: %{run} %t.out
