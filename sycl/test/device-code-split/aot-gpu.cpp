// REQUIRES: ocloc, gpu

// RUN: %clangxx -fsycl -fsycl-device-code-split=per_source -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xsycl-target-backend=spir64_gen-unknown-unknown-sycldevice "-device skl" -I %S/Inputs -o %t.out %S/split-per-source-main.cpp %S/Inputs/split-per-source-second-file.cpp
// RUN: %GPU_RUN_PLACEHOLDER %t.out
