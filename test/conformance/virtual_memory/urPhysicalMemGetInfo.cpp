// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>
#include <uur/known_failure.h>

using urPhysicalMemGetInfoTest = uur::urPhysicalMemTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urPhysicalMemGetInfoTest);

TEST_P(urPhysicalMemGetInfoTest, Context) {
    UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

    size_t info_size = 0;

    ASSERT_SUCCESS(urPhysicalMemGetInfo(
        physical_mem, UR_PHYSICAL_MEM_INFO_CONTEXT, 0, nullptr, &info_size));
    ASSERT_NE(info_size, 0);

    ur_context_handle_t returned_context = nullptr;
    ASSERT_SUCCESS(urPhysicalMemGetInfo(physical_mem,
                                        UR_PHYSICAL_MEM_INFO_CONTEXT, info_size,
                                        &returned_context, nullptr));

    ASSERT_EQ(context, returned_context);
}

TEST_P(urPhysicalMemGetInfoTest, Device) {
    UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

    size_t info_size = 0;

    ASSERT_SUCCESS(urPhysicalMemGetInfo(
        physical_mem, UR_PHYSICAL_MEM_INFO_DEVICE, 0, nullptr, &info_size));
    ASSERT_NE(info_size, 0);

    ur_device_handle_t returned_device = nullptr;
    ASSERT_SUCCESS(urPhysicalMemGetInfo(physical_mem,
                                        UR_PHYSICAL_MEM_INFO_DEVICE, info_size,
                                        &returned_device, nullptr));

    ASSERT_EQ(device, returned_device);
}

TEST_P(urPhysicalMemGetInfoTest, Size) {
    UUR_KNOWN_FAILURE_ON(uur::LevelZero{});

    size_t info_size = 0;

    ASSERT_SUCCESS(urPhysicalMemGetInfo(physical_mem, UR_PHYSICAL_MEM_INFO_SIZE,
                                        0, nullptr, &info_size));
    ASSERT_NE(info_size, 0);

    size_t returned_size = 0;
    ASSERT_SUCCESS(urPhysicalMemGetInfo(physical_mem, UR_PHYSICAL_MEM_INFO_SIZE,
                                        info_size, &returned_size, nullptr));

    ASSERT_EQ(size, returned_size);
}

TEST_P(urPhysicalMemGetInfoTest, Properties) {
    UUR_KNOWN_FAILURE_ON(uur::LevelZero{});
    size_t info_size = 0;

    ASSERT_SUCCESS(urPhysicalMemGetInfo(
        physical_mem, UR_PHYSICAL_MEM_INFO_PROPERTIES, 0, nullptr, &info_size));
    ASSERT_NE(info_size, 0);

    ur_physical_mem_properties_t returned_properties = {};
    ASSERT_SUCCESS(
        urPhysicalMemGetInfo(physical_mem, UR_PHYSICAL_MEM_INFO_PROPERTIES,
                             info_size, &returned_properties, nullptr));

    ASSERT_EQ(properties.stype, returned_properties.stype);
    ASSERT_EQ(properties.pNext, returned_properties.pNext);
    ASSERT_EQ(properties.flags, returned_properties.flags);
}

TEST_P(urPhysicalMemGetInfoTest, ReferenceCount) {
    size_t info_size = 0;

    ASSERT_SUCCESS(urPhysicalMemGetInfo(physical_mem,
                                        UR_PHYSICAL_MEM_INFO_REFERENCE_COUNT, 0,
                                        nullptr, &info_size));
    ASSERT_NE(info_size, 0);

    uint32_t returned_reference_count = 0;
    ASSERT_SUCCESS(
        urPhysicalMemGetInfo(physical_mem, UR_PHYSICAL_MEM_INFO_REFERENCE_COUNT,
                             info_size, &returned_reference_count, nullptr));

    ASSERT_EQ(returned_reference_count, 1);
}
