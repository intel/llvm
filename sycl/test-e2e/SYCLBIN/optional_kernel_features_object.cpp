//==--- optional_kernel_features_object.cpp --- SYCLBIN extension tests
//-----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aspect-usm_device_allocations

// -- Test for compiling and loading a kernel bundle with a SYCLBIN containing
//    the use of optional kernel features.

// UNSUPPORTED: hip
// UNSUPPORTED-INTENDED: HIP backend does not implement linking.

// RUN: %clangxx --offload-new-driver -fsyclbin=object %{sycl_target_opts} %S/Inputs/optional_kernel_features.cpp -o %t.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.syclbin

#define SYCLBIN_OBJECT_STATE

#include "Inputs/optional_kernel_features.hpp"
