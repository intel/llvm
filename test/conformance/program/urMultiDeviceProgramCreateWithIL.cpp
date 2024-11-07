
// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/raii.h>

using urMultiDeviceProgramTest = uur::urMultiDeviceProgramTest;

// Test binary sizes and binaries obtained from urProgramGetInfo when program is built for a subset of devices in the context.
TEST_F(urMultiDeviceProgramTest, urMultiDeviceProgramGetInfo) {
    // Run test only for level zero backend which supports urProgramBuildExp.
    ur_platform_backend_t backend;
    ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                     sizeof(backend), &backend, nullptr));
    if (backend != UR_PLATFORM_BACKEND_LEVEL_ZERO) {
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
    ASSERT_SUCCESS(
        urProgramBuildExp(program, subset.size(), subset.data(), nullptr));

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
    for (size_t i = associated_devices.size() / 2;
         i < associated_devices.size(); i++) {
        ASSERT_EQ(binary_sizes[i], 0);
        pointers[i] = binaries[i].data();
    }

    ASSERT_SUCCESS(urProgramGetInfo(program, UR_PROGRAM_INFO_BINARIES,
                                    sizeof(uint8_t *) * pointers.size(),
                                    pointers.data(), nullptr));
    for (size_t i = 0; i < associated_devices.size() / 2; i++) {
        ASSERT_NE(binaries[i].size(), 0);
    }
    for (size_t i = associated_devices.size() / 2;
         i < associated_devices.size(); i++) {
        ASSERT_EQ(binaries[i].size(), 0);
    }
}
