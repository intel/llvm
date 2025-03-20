// REQUIRES: ocloc, gpu, target-spir

// XFAIL: windows && !(build-mode && run-mode)
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/17553

// RUN: %clangxx -fsycl -fsycl-device-code-split=per_source \
// RUN:   -fsycl-targets=spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen \
// RUN:   "-device tgllp" -I %S/Inputs -o %t.out \
// RUN:   %S/split-per-source-main.cpp \
// RUN:   %S/Inputs/split-per-source-second-file.cpp \
// RUN:   -fsycl-dead-args-optimization
// RUN: %{run} %t.out
