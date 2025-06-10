// REQUIRES: cuda || hip
// RUN: %{build} -DBUILD_FOR_GPU -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

//==- attributes_cuda_hip.cpp - SYCL sub_group attributes test -*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: sg-32

#include "attributes_helper.hpp"

int main() { return runTests(); }
