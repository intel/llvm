// REQUIRES: ocloc, gpu
// UNSUPPORTED: cuda
// CUDA does neither support device code splitting nor SPIR.
// Issue #128: The test fails for driver 27.20.100.9168
// XFAIL: windows
//
// RUN: %clangxx -fsycl -fsycl-device-code-split=per_source \
// RUN:   -fsycl-targets=spir64_gen-unknown-unknown-sycldevice \
// RUN:   -Xsycl-target-backend=spir64_gen-unknown-unknown-sycldevice \
// RUN:   "-device *" -I %S/Inputs -o %t.out \
// RUN:   %S/split-per-source-main.cpp \
// RUN:   %S/Inputs/split-per-source-second-file.cpp
// RUN: %GPU_RUN_PLACEHOLDER %t.out
