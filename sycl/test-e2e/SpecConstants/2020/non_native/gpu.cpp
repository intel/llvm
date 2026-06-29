// REQUIRES: ocloc, gpu, target-spir

// UNSUPPORTED: linux && arch-intel_gpu_pvc
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/22405

// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %S/Inputs/common.cpp -o %t.out -fsycl-dead-args-optimization
// RUN: %{run} %t.out

// This test checks correctness of SYCL2020 non-native specialization constants
// on GPU device
