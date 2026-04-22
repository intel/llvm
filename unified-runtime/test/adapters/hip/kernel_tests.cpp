// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "kernel.hpp"
#include "uur/fixtures.h"
#include "uur/raii.h"

using hipKernelTest = uur::urKernelExecutionTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(hipKernelTest);

TEST_P(hipKernelTest, URKernelLaunchWithArgsExpLargeArgFails) {
  // The HIP adapter can't do proper argument validation so any kernel will
  // work for this test.
  std::array<uint8_t, 4004> data;
  data.fill(0);

  ur_exp_kernel_arg_value_t arg_val = {};
  arg_val.value = data.data();
  ur_exp_kernel_arg_properties_t arg = {
      UR_STRUCTURE_TYPE_EXP_KERNEL_ARG_PROPERTIES,
      nullptr,
      UR_EXP_KERNEL_ARG_TYPE_VALUE,
      0,
      4004,
      arg_val};

  size_t offset = 0;
  size_t global_size = 1;
  size_t local_size = 1;
  ASSERT_EQ_RESULT(urEnqueueKernelLaunchWithArgsExp(
                       queue, kernel, 1, &offset, &global_size, &local_size, 1,
                       &arg, nullptr, 0, nullptr, nullptr),
                   UR_RESULT_ERROR_OUT_OF_RESOURCES);
}
