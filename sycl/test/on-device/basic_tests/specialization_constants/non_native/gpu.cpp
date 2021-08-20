// REQUIRES: ocloc, gpu, TEMPORARY_DISABLED
// UNSUPPORTED: cuda
// CUDA is not compatible with SPIR.
//
// UNSUPPORTED: rocm_nvidia
// UNSUPPORTED: rocm_amd
// ROCm is not compatible with SPIR.

// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xsycl-target-backend=spir64_gen-unknown-unknown-sycldevice "-device *" %S/Inputs/common.cpp -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test checks correctness of SYCL2020 non-native specialization constants
// on GPU device
