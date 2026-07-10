// REQUIRES: ocloc, gpu, target-spir, !gpu-intel-gen12
//
// RUN: %clangxx -fsycl -fsycl-device-code-split=per_source \
// RUN:   -fsycl-targets=intel_gpu_dg2 \
// RUN:   -I %S/Inputs -o %t.out \
// RUN:   %S/split-per-source-main.cpp \
// RUN:   %S/Inputs/split-per-source-second-file.cpp \
// RUN:   -fsycl-dead-args-optimization
// RUN: %{run} %t.out
