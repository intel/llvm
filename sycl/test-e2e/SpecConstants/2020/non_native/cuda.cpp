// REQUIRES: target-nvidia

// RUN: %clangxx -fsycl %{sycl_target_opts} %S/Inputs/common.cpp -o %t.out
// RUN: %{run-unfiltered-devices} env ONEAPI_DEVICE_SELECTOR="cuda:*" %t.out

// This test checks correctness of SYCL2020 non-native specialization constants
// on CUDA device
