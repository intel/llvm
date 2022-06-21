// REQUIRES: cuda

// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda %S/Inputs/common.cpp -o %t.out
// RUN: env SYCL_DEVICE_FILTER=cuda %t.out

// This test checks correctness of SYCL2020 non-native specialization constants
// on CUDA device
