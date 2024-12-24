// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

using urPhysicalMemGetInfoTest = uur::urPhysicalMemTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urPhysicalMemGetInfoTest);

TEST_P(urPhysicalMemGetInfoTest, Context) {
    size_t info_size = 0;

    ASSERT_SUCCESS(urPhysicalMemGetInfo(
        physical_mem, UR_PHYSICAL_MEM_INFO_CONTEXT, 0, nullptr, &info_size));
    ASSERT_NE(info_size, 0);

    std::vector<uint8_t> data(info_size);
    ASSERT_SUCCESS(urPhysicalMemGetInfo(physical_mem,
                                        UR_PHYSICAL_MEM_INFO_CONTEXT,
                                        data.size(), data.data(), nullptr));

    auto returned_context =
        reinterpret_cast<ur_context_handle_t *>(data.data());
    ASSERT_EQ(context, *returned_context);
}

TEST_P(urPhysicalMemGetInfoTest, Device) {
    size_t info_size = 0;

    ASSERT_SUCCESS(urPhysicalMemGetInfo(
        physical_mem, UR_PHYSICAL_MEM_INFO_DEVICE, 0, nullptr, &info_size));
    ASSERT_NE(info_size, 0);

    std::vector<uint8_t> data(info_size);
    ASSERT_SUCCESS(urPhysicalMemGetInfo(physical_mem,
                                        UR_PHYSICAL_MEM_INFO_DEVICE,
                                        data.size(), data.data(), nullptr));

    auto returned_device = reinterpret_cast<ur_device_handle_t *>(data.data());
    ASSERT_EQ(device, *returned_device);
}

TEST_P(urPhysicalMemGetInfoTest, Size) {
    size_t info_size = 0;

    ASSERT_SUCCESS(urPhysicalMemGetInfo(physical_mem, UR_PHYSICAL_MEM_INFO_SIZE,
                                        0, nullptr, &info_size));
    ASSERT_NE(info_size, 0);

    std::vector<uint8_t> data(info_size);
    ASSERT_SUCCESS(urPhysicalMemGetInfo(physical_mem, UR_PHYSICAL_MEM_INFO_SIZE,
                                        data.size(), data.data(), nullptr));

    auto returned_size = reinterpret_cast<size_t *>(data.data());
    ASSERT_EQ(size, *returned_size);
}

TEST_P(urPhysicalMemGetInfoTest, Properties) {
    size_t info_size = 0;

    ASSERT_SUCCESS(urPhysicalMemGetInfo(
        physical_mem, UR_PHYSICAL_MEM_INFO_PROPERTIES, 0, nullptr, &info_size));
    ASSERT_NE(info_size, 0);

    std::vector<uint8_t> data(info_size);
    ASSERT_SUCCESS(urPhysicalMemGetInfo(physical_mem,
                                        UR_PHYSICAL_MEM_INFO_PROPERTIES,
                                        data.size(), data.data(), nullptr));

    auto returned_properties =
        reinterpret_cast<ur_physical_mem_properties_t *>(data.data());
    ASSERT_EQ(properties.stype, returned_properties->stype);
    ASSERT_EQ(properties.pNext, returned_properties->pNext);
    ASSERT_EQ(properties.flags, returned_properties->flags);
}

TEST_P(urPhysicalMemGetInfoTest, ReferenceCount) {
    size_t info_size = 0;

    ASSERT_SUCCESS(urPhysicalMemGetInfo(physical_mem,
                                        UR_PHYSICAL_MEM_INFO_REFERENCE_COUNT, 0,
                                        nullptr, &info_size));
    ASSERT_NE(info_size, 0);

    std::vector<uint8_t> data(info_size);
    ASSERT_SUCCESS(urPhysicalMemGetInfo(physical_mem,
                                        UR_PHYSICAL_MEM_INFO_REFERENCE_COUNT,
                                        data.size(), data.data(), nullptr));

    const size_t ReferenceCount =
        *reinterpret_cast<const uint32_t *>(data.data());
    ASSERT_EQ(ReferenceCount, 1);
}
