// UNSUPPORTED: hip
// RUN: %if cuda %{%clangxx -fsycl -fsycl-targets=%sycl_triple -Xsycl-target-backend --cuda-gpu-arch=sm_80 %s -o %t.out %}
// TODO enable the below when CI supports >=sm_80
// RUNx: %if cuda %{%GPU_RUN_PLACEHOLDER %t.out %}
// RUN: %clangxx -fsycl %s -o %t.out
// TODO currently the feature isn't supported on FPGA.
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUNx: %ACC_RUN_PLACEHOLDER %t.out

//==----------- bfloat16_type.cpp - SYCL bfloat16 type test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bfloat16_type.hpp"

int main() { return run_tests(); }
