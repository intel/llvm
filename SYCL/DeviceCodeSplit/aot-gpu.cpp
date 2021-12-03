// REQUIRES: ocloc, gpu
// UNSUPPORTED: cuda || hip
// CUDA does neither support device code splitting nor SPIR.
//
// RUN: %clangxx -fsycl -fsycl-device-code-split=per_source \
// RUN:   -fsycl-targets=spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen \
// RUN:   %gpu_aot_target_opts -I %S/Inputs -o %t.out \
// RUN:   %S/split-per-source-main.cpp \
// RUN:   %S/Inputs/split-per-source-second-file.cpp \
// RUN:   -fsycl-dead-args-optimization
// RUN: %GPU_RUN_PLACEHOLDER %t.out
