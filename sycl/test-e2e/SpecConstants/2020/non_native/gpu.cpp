// REQUIRES: ocloc, gpu, target-spir

// UNSUPPORTED: linux && arch-intel_gpu_pvc
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/22405

// RUN: %clangxx -fsycl -fsycl-targets=%{intel_gpu_aot_targets} %S/Inputs/common.cpp -o %t.out -fsycl-dead-args-optimization
// RUN: %{run} %t.out

// This test checks correctness of SYCL2020 non-native specialization constants
// on GPU device
