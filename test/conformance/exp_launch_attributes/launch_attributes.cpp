// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include <uur/fixtures.h>

struct urEnqueueKernelLaunchCustomTest : uur::urKernelExecutionTest {
  void SetUp() override {
    program_name = "fill";
    UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
  }

  uint32_t val = 42;
  size_t global_size = 32;
  size_t global_offset = 0;
  size_t n_dimensions = 1;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueKernelLaunchCustomTest);

TEST_P(urEnqueueKernelLaunchCustomTest, Success) {

  size_t returned_size;
  ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_EXTENSIONS, 0, nullptr,
                                 &returned_size));

  std::unique_ptr<char[]> returned_extensions(new char[returned_size]);

  ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_EXTENSIONS,
                                 returned_size, returned_extensions.get(),
                                 nullptr));

  std::string_view extensions_string(returned_extensions.get());
  const bool launch_attributes_support =
      extensions_string.find(UR_LAUNCH_ATTRIBUTES_EXTENSION_STRING_EXP) !=
      std::string::npos;

  if (!launch_attributes_support) {
    GTEST_SKIP() << "EXP launch attributes feature is not supported.";
  }

  ur_mem_handle_t buffer = nullptr;
  AddBuffer1DArg(sizeof(val) * global_size, &buffer);
  AddPodArg(val);

  ur_exp_launch_attribute_t attr[1];
  attr[0].id = UR_EXP_LAUNCH_ATTRIBUTE_ID_CLUSTER_DIMENSION;
  attr[0].value.clusterDim[0] = global_size;
  attr[0].value.clusterDim[1] = 1;
  attr[0].value.clusterDim[2] = 1;

  ASSERT_SUCCESS(urEnqueueKernelLaunchCustomExp(queue, kernel, n_dimensions,
                                                &global_size, nullptr, 1,
                                                attr, 0, nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));
  ValidateBuffer(buffer, sizeof(val) * global_size, val);
}
