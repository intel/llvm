// REQUIRES: ocloc, gpu, target-spir

// RUN: %clangxx -fsycl -fsycl-targets=%{gpu_aot_target} %S/Inputs/common.cpp -o %t.out -fsycl-dead-args-optimization
// RUN: %{run} %t.out

// This test checks correctness of SYCL2020 non-native specialization constants
// on GPU device
