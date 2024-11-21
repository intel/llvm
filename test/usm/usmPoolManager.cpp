// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ur_pool_manager.hpp"

#include <uur/fixtures.h>

using urUsmPoolDescriptorTest = uur::urMultiDeviceContextTest;

UUR_INSTANTIATE_PLATFORM_TEST_SUITE_P(urUsmPoolDescriptorTest);

auto createMockPoolHandle() {
    static uintptr_t uniqueAddress = 0x1;
    return umf::pool_unique_handle_t(
        (umf_memory_pool_handle_t)(uniqueAddress++),
        [](umf_memory_pool_t *) {});
}

TEST_P(urUsmPoolDescriptorTest, poolIsPerContextTypeAndDevice) {
    auto &devices = uur::DevicesEnvironment::instance->devices;

    auto [ret, pool_descriptors] =
        usm::pool_descriptor::create(nullptr, this->context);
    ASSERT_EQ(ret, UR_RESULT_SUCCESS);

    size_t hostPools = 0;
    size_t devicePools = 0;
    size_t sharedPools = 0;

    for (auto &desc : pool_descriptors) {
        ASSERT_EQ(desc.poolHandle, nullptr);
        ASSERT_EQ(desc.hContext, this->context);

        if (desc.type == UR_USM_TYPE_DEVICE) {
            devicePools++;
        } else if (desc.type == UR_USM_TYPE_SHARED) {
            sharedPools++;
        } else if (desc.type == UR_USM_TYPE_HOST) {
            hostPools += 2;
        } else {
            FAIL();
        }
    }

    // Each device has pools for Host, Device, Shared, SharedReadOnly only
    ASSERT_EQ(pool_descriptors.size(), 4 * devices.size());
    ASSERT_EQ(hostPools, 1);
    ASSERT_EQ(devicePools, devices.size());
    ASSERT_EQ(sharedPools, devices.size() * 2);
}

// TODO: add test with sub-devices

struct urUsmPoolManagerTest : public uur::urContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
        auto [ret, descs] = usm::pool_descriptor::create(nullptr, context);
        ASSERT_EQ(ret, UR_RESULT_SUCCESS);
        poolDescriptors = std::move(descs);
    }

    std::vector<usm::pool_descriptor> poolDescriptors;
};

TEST_P(urUsmPoolManagerTest, poolManagerPopulate) {
    auto [ret, manager] = usm::pool_manager<usm::pool_descriptor>::create();
    ASSERT_EQ(ret, UR_RESULT_SUCCESS);

    for (auto &desc : poolDescriptors) {
        // Populate the pool manager
        auto poolUnique = createMockPoolHandle();
        ASSERT_NE(poolUnique, nullptr);
        ret = manager.addPool(desc, std::move(poolUnique));
        ASSERT_EQ(ret, UR_RESULT_SUCCESS);
    }

    for (auto &desc : poolDescriptors) {
        // Confirm that there is a pool for each descriptor
        auto hPoolOpt = manager.getPool(desc);
        ASSERT_TRUE(hPoolOpt.has_value());
        ASSERT_NE(hPoolOpt.value(), nullptr);
    }
}

TEST_P(urUsmPoolManagerTest, poolManagerInsertExisting) {
    auto [ret, manager] = usm::pool_manager<usm::pool_descriptor>::create();
    ASSERT_EQ(ret, UR_RESULT_SUCCESS);

    const auto &desc = poolDescriptors[0];

    auto poolUnique = createMockPoolHandle();
    ASSERT_NE(poolUnique, nullptr);

    ret = manager.addPool(desc, std::move(poolUnique));
    ASSERT_EQ(ret, UR_RESULT_SUCCESS);

    // Inserting an existing key should return an error
    ret = manager.addPool(desc, createMockPoolHandle());
    ASSERT_EQ(ret, UR_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(urUsmPoolManagerTest, poolManagerGetNonexistant) {
    auto [ret, manager] = usm::pool_manager<usm::pool_descriptor>::create();
    ASSERT_EQ(ret, UR_RESULT_SUCCESS);

    for (auto &desc : poolDescriptors) {
        auto hPool = manager.getPool(desc);
        ASSERT_FALSE(hPool.has_value());
    }
}

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urUsmPoolManagerTest);
