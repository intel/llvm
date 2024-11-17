// REQUIRES: ocloc, gpu
// UNSUPPORTED: cuda || hip
// CUDA does neither support device code splitting nor SPIR.
// Test with `--offload-new-driver`
//
// RUN: %clangxx -fsycl -fsycl-device-code-split=per_source \
// RUN:   -fsycl-targets=spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen \
// RUN:   "-device tgllp" -I %S/Inputs -o %t.out \
// RUN:   %S/split-per-source-main.cpp \
// RUN:   %S/Inputs/split-per-source-second-file.cpp \
// RUN:   -fsycl-dead-args-optimization --offload-new-driver
// RUN: %{run} %t.out
