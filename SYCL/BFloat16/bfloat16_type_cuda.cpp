// REQUIRES: gpu, cuda
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -Xsycl-target-backend --cuda-gpu-arch=sm_80 %s -o %t.out
// TODO: Currently the CI does not have a sm_80 capable machine. Enable the test
// execution once it does.
// RUNx: %t.out

//==--------- bfloat16_type_cuda.cpp - SYCL bfloat16 type test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bfloat16_type.hpp"

int main() { return run_tests(); }
