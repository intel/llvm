
// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/raii.h>

using urMultiDeviceProgramTest = uur::urMultiDeviceProgramTest;
UUR_INSTANTIATE_PLATFORM_TEST_SUITE(urMultiDeviceProgramTest);

// Test binary sizes and binaries obtained from urProgramGetInfo when program is
// built for a subset of devices in the context.
TEST_P(urMultiDeviceProgramTest, urMultiDeviceProgramGetInfo) {
  // Run test only for level zero backend which supports urProgramBuildExp.
  ur_backend_t backend;
  ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                   sizeof(backend), &backend, nullptr));
  if (backend != UR_BACKEND_LEVEL_ZERO) {
    GTEST_SKIP();
  }

  std::vector<ur_device_handle_t> associated_devices(devices.size());
  ASSERT_SUCCESS(
      urProgramGetInfo(program, UR_PROGRAM_INFO_DEVICES,
                       associated_devices.size() * sizeof(ur_device_handle_t),
                       associated_devices.data(), nullptr));

  // Build program for the first half of devices.
  auto subset = std::vector<ur_device_handle_t>(
      associated_devices.begin(),
      associated_devices.begin() + associated_devices.size() / 2);
  ASSERT_SUCCESS(urProgramBuildExp(program, subset.size(), subset.data(),
                                   ur_exp_program_flags_t{}, nullptr));

  std::vector<size_t> binary_sizes(associated_devices.size());
  ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_BINARY_SIZES,
                                  binary_sizes.size() * sizeof(size_t),
                                  binary_sizes.data(), nullptr));

  std::vector<std::vector<char>> binaries(associated_devices.size());
  std::vector<char *> pointers(associated_devices.size());
  for (size_t i = 0; i < associated_devices.size() / 2; i++) {
    ASSERT_NE(binary_sizes[i], 0);
    binaries[i].resize(binary_sizes[i]);
    pointers[i] = binaries[i].data();
  }
  for (size_t i = associated_devices.size() / 2; i < associated_devices.size();
       i++) {
    ASSERT_EQ(binary_sizes[i], 0);
    pointers[i] = binaries[i].data();
  }

  ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_BINARIES,
                                  sizeof(uint8_t *) * pointers.size(),
                                  pointers.data(), nullptr));
  for (size_t i = 0; i < associated_devices.size() / 2; i++) {
    ASSERT_NE(binaries[i].size(), 0);
  }
  for (size_t i = associated_devices.size() / 2; i < associated_devices.size();
       i++) {
    ASSERT_EQ(binaries[i].size(), 0);
  }
}

// Build program for the second device only and check validity of the binary returned by urProgramGetInfo
// by recreating program from the binary and building it.
TEST_P(urMultiDeviceProgramTest, urMultiDeviceProgramGetInfoBinaries) {
  ur_backend_t backend;
  ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                   sizeof(backend), &backend, nullptr));
  if (backend != UR_BACKEND_LEVEL_ZERO) {
    GTEST_SKIP();
  }
  std::vector<ur_device_handle_t> associated_devices(devices.size());
  ASSERT_SUCCESS(
      urProgramGetInfo(program, UR_PROGRAM_INFO_DEVICES,
                       associated_devices.size() * sizeof(ur_device_handle_t),
                       associated_devices.data(), nullptr));
  if (associated_devices.size() < 2) {
    GTEST_SKIP();
  }

  // Build program for the second device only.
  ASSERT_SUCCESS(urProgramBuildExp(program, 1, associated_devices.data() + 1,
                                   ur_exp_program_flags_t{}, nullptr));
  std::vector<size_t> binary_sizes(associated_devices.size());
  ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_BINARY_SIZES,
                                  binary_sizes.size() * sizeof(size_t),
                                  binary_sizes.data(), nullptr));
  std::vector<std::vector<uint8_t>> binaries(associated_devices.size());
  std::vector<const uint8_t *> pointers(associated_devices.size());
  for (size_t i = 0; i < associated_devices.size(); i++) {
    binaries[i].resize(binary_sizes[i]);
    pointers[i] = binaries[i].data();
  }

  ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_BINARIES,
                                  sizeof(uint8_t *) * pointers.size(),
                                  pointers.data(), nullptr));

  // Now create program from the obtained binary and build to check validity.
  ur_program_handle_t program_from_binary = nullptr;
  ASSERT_SUCCESS(urProgramCreateWithBinary(
      context, 1, associated_devices.data() + 1, binary_sizes.data() + 1,
      pointers.data() + 1, nullptr, &program_from_binary));
  ASSERT_NE(program_from_binary, nullptr);
  ASSERT_SUCCESS(urProgramBuildExp(program_from_binary, 1,
                                   associated_devices.data() + 1,
                                   ur_exp_program_flags_t{}, nullptr));
  ASSERT_SUCCESS(urProgramRelease(program_from_binary));
}
