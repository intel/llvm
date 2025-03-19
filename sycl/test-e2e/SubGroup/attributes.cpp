// UNSUPPORTED: target-amd, target-nvidia
// UNSUPPORTED-INTENDED: This test is not meant to be run on CUDA/HIP. Instead
// `attributes_cuda_hip.cpp` is designed to test those backends. This is needed
// as the CI is set up such that it only builds a test once for all available
// devices, this is not suitable, as GPU targets will compile-time-check the
// sub-group size and error out if it is not correct.

// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

//==------- attributes.cpp - SYCL sub_group attributes test ----*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "attributes_helper.hpp"

int main() { return runTests(); }
