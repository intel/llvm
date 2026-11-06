// REQUIRES: ocloc, gpu, spir-family

// RUN: %clangxx -fsycl %aot_options %S/Inputs/common.cpp -o %t.out -fsycl-dead-args-optimization
// RUN: %{run} %t.out

// This test checks correctness of SYCL2020 non-native specialization constants
// on GPU device
