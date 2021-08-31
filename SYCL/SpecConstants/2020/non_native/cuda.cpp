// REQUIRES: cuda

// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-unknown-unknown-sycldevice %S/Inputs/common.cpp -o %t.out
// RUN: env SYCL_DEVICE_FILTER=cuda %t.out

// TODO: enable this test then compile-time error in sycl-post-link is fixed
// UNSUPPORTED: cuda

// This test checks correctness of SYCL2020 non-native specialization constants
// on CUDA device
