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

TEST_P(hipKernelTest, URKernelArgumentLarge) {
  // The HIP adapter can't do proper argument validation so any kernel will
  // work for this test.
  std::array<uint8_t, 4004> data;
  data.fill(0);
  ASSERT_EQ_RESULT(urKernelSetArgValue(kernel, 0, 4004, nullptr, data.data()),
                   UR_RESULT_ERROR_OUT_OF_RESOURCES);
}
