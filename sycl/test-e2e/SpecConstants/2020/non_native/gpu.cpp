// REQUIRES: ocloc, gpu, target-spir
// REQUIRES: intel-gpu-aot-targets || !new-offload-model

// UNSUPPORTED: linux && arch-intel_gpu_pvc
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/22405

// RUN: %clangxx -fsycl %{gpu_aot_opts} %S/Inputs/common.cpp -o %t.out -fsycl-dead-args-optimization
// RUN: %{run} %t.out

// This test checks correctness of SYCL2020 non-native specialization constants
// on GPU device
