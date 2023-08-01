// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../unified_malloc_framework/common/pool.hpp"
#include "../unified_malloc_framework/common/provider.hpp"
#include "ur_pool_manager.hpp"

#include <uur/fixtures.h>

#include <unordered_set>

struct urUsmPoolManagerTest
    : public uur::urMultiDeviceContextTest,
      ::testing::WithParamInterface<ur_usm_pool_handle_t> {};

TEST_P(urUsmPoolManagerTest, poolIsPerContextTypeAndDevice) {
    auto &devices = uur::DevicesEnvironment::instance->devices;
    auto poolHandle = this->GetParam();

    auto [ret, pool_descriptors] =
        usm::pool_descriptor::create(poolHandle, this->context);
    ASSERT_EQ(ret, UR_RESULT_SUCCESS);

    size_t hostPools = 0;
    size_t devicePools = 0;
    size_t sharedPools = 0;

    for (auto &desc : pool_descriptors) {
        ASSERT_EQ(desc.poolHandle, poolHandle);
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

INSTANTIATE_TEST_SUITE_P(urUsmPoolManagerTest, urUsmPoolManagerTest,
                         ::testing::Values(nullptr));

// TODO: add test with sub-devices
