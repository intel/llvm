// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ur_print.hpp"
#include "uur/fixtures.h"
#include "uur/raii.h"
#include "uur/utils.h"

#include <map>
#include <string>

using urMemoryResidencyTest = uur::urMultiDeviceContextTestTemplate<1>;

TEST_F(urMemoryResidencyTest, allocatingDeviceMemoryWillResultInOOM) {
    static constexpr size_t allocSize = 1024 * 1024;

    if (!uur::isPVC(uur::DevicesEnvironment::instance->devices[0])) {
        GTEST_SKIP() << "Test requires a PVC device";
    }

    size_t initialMemFree = 0;
    ASSERT_SUCCESS(
        urDeviceGetInfo(uur::DevicesEnvironment::instance->devices[0],
                        UR_DEVICE_INFO_GLOBAL_MEM_FREE, sizeof(size_t),
                        &initialMemFree, nullptr));

    if (initialMemFree < allocSize) {
        GTEST_SKIP() << "Not enough device memory available";
    }

    void *ptr = nullptr;
    ASSERT_SUCCESS(
        urUSMDeviceAlloc(context, uur::DevicesEnvironment::instance->devices[0],
                         nullptr, nullptr, allocSize, &ptr));

    size_t currentMemFree = 0;
    ASSERT_SUCCESS(
        urDeviceGetInfo(uur::DevicesEnvironment::instance->devices[0],
                        UR_DEVICE_INFO_GLOBAL_MEM_FREE, sizeof(size_t),
                        &currentMemFree, nullptr));

    // amount of free memory should decrease after making a memory allocation resident
    ASSERT_LE(currentMemFree, initialMemFree);

    ASSERT_SUCCESS(urUSMFree(context, ptr));
}
