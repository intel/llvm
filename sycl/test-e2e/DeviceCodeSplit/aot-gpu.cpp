// REQUIRES: ocloc, gpu, target-spir, !gpu-intel-gen12
//
// RUN: %clangxx -fsycl -fsycl-device-code-split=per_source \
// RUN:   -fsycl-targets=spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen \
// RUN:   "-device dg2" -I %S/Inputs -o %t.out \
// RUN:   %S/split-per-source-main.cpp \
// RUN:   %S/Inputs/split-per-source-second-file.cpp \
// RUN:   -fsycl-dead-args-optimization
// RUN: %{run} %t.out
