// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

  std::vector<ur_exp_launch_attribute_t> attrs(1);
  attrs[0].id = UR_EXP_LAUNCH_ATTRIBUTE_ID_IGNORE;

  ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_PROFILE, 0, nullptr,
                                 &returned_size));

  std::unique_ptr<char[]> returned_backend(new char[returned_size]);

  ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_PROFILE, returned_size,
                                 returned_backend.get(), nullptr));

  std::string_view backend_string(returned_backend.get());
  const bool cuda_backend = backend_string.find("CUDA") != std::string::npos;

  if (cuda_backend) {
    ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_VERSION, 0, nullptr,
                                   &returned_size));

    std::unique_ptr<char[]> returned_compute_capability(
        new char[returned_size]);

    ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_VERSION,
                                   returned_size,
                                   returned_compute_capability.get(), nullptr));

    auto compute_capability =
        std::stof(std::string(returned_compute_capability.get()));

    if (compute_capability >= 6.0) {
      ur_exp_launch_attribute_t coop_attr;
      coop_attr.id = UR_EXP_LAUNCH_ATTRIBUTE_ID_COOPERATIVE;
      coop_attr.value.cooperative = 1;
      attrs.push_back(coop_attr);
    }

    if (compute_capability >= 9.0) {
      ur_exp_launch_attribute_t cluster_dims_attr;
      cluster_dims_attr.id = UR_EXP_LAUNCH_ATTRIBUTE_ID_CLUSTER_DIMENSION;
      cluster_dims_attr.value.clusterDim[0] = 32;
      cluster_dims_attr.value.clusterDim[1] = 1;
      cluster_dims_attr.value.clusterDim[2] = 1;

      attrs.push_back(cluster_dims_attr);
    }
  }
  ur_mem_handle_t buffer = nullptr;
  AddBuffer1DArg(sizeof(val) * global_size, &buffer);
  AddPodArg(val);

  ASSERT_SUCCESS(urEnqueueKernelLaunchCustomExp(
      queue, kernel, n_dimensions, &global_size, nullptr, 1, &attrs[0], 0,
      nullptr, nullptr));
  ASSERT_SUCCESS(urQueueFinish(queue));
  ValidateBuffer(buffer, sizeof(val) * global_size, val);
}
