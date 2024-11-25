// UNSUPPORTED: true
// REQUIRES: ocloc, gpu
// UNSUPPORTED: cuda
// CUDA is not compatible with SPIR.
//
// UNSUPPORTED: hip
// HIP is not compatible with SPIR.

// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %S/Inputs/common.cpp -o %t.out -fsycl-dead-args-optimization
// RUN: %{run} %t.out

// This test checks correctness of SYCL2020 non-native specialization constants
// on GPU device
