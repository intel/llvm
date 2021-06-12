// REQUIRES: ocloc, gpu
// UNSUPPORTED: cuda
// CUDA is not compatible with SPIR.

// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen-unknown-unknown -Xsycl-target-backend=spir64_gen-unknown-unknown "-device *" %S/Inputs/common.cpp -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test checks correctness of SYCL2020 non-native specialization constants
// on GPU device
