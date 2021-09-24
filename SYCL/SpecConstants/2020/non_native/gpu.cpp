// REQUIRES: ocloc, gpu, TEMPORARY_DISABLED
// UNSUPPORTED: cuda
// CUDA is not compatible with SPIR.
//
// UNSUPPORTED: hip
// HIP is not compatible with SPIR.

// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device *" %S/Inputs/common.cpp -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test checks correctness of SYCL2020 non-native specialization constants
// on GPU device
