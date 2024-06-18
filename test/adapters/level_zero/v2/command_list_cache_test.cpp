// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "command_list_cache.hpp"
#include "common.hpp"

#include "context.hpp"
#include "device.hpp"

#include "uur/fixtures.h"

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <unordered_set>

struct CommandListCacheTest : public uur::urContextTest {};

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(CommandListCacheTest);

TEST_P(CommandListCacheTest, CanStoreAndRetriveImmediateAndRegularCmdLists) {
    v2::command_list_cache_t cache(context->ZeContext);

    bool IsInOrder = false;
    uint32_t Ordinal = 0;

    ze_command_queue_mode_t Mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    ze_command_queue_priority_t Priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

    static constexpr int numListsPerType = 3;
    std::unordered_set<ze_command_list_handle_t> regCmdLists;
    std::unordered_set<ze_command_list_handle_t> immCmdLists;

    // get command lists from the cache
    for (int i = 0; i < numListsPerType; ++i) {
        auto [it, _] = regCmdLists.emplace(
            cache.getRegularCommandList(device->ZeDevice, IsInOrder, Ordinal)
                .release());
        ASSERT_TRUE(*it != nullptr);

        std::tie(it, _) = immCmdLists.emplace(
            cache
                .getImmediateCommandList(device->ZeDevice, IsInOrder, Ordinal,
                                         Mode, Priority)
                .release());
        ASSERT_TRUE(*it != nullptr);
    }

    // store them back into the cache
    for (auto cmdList : regCmdLists) {
        cache.addRegularCommandList(
            v2::raii::ze_command_list_t(cmdList, &zeCommandListDestroy),
            device->ZeDevice, IsInOrder, Ordinal);
    }
    for (auto cmdList : immCmdLists) {
        cache.addImmediateCommandList(
            v2::raii::ze_command_list_t(cmdList, &zeCommandListDestroy),
            device->ZeDevice, IsInOrder, Ordinal, Mode, Priority);
    }

    // verify we get back the same command lists
    for (int i = 0; i < numListsPerType; ++i) {
        auto regCmdList =
            cache.getRegularCommandList(device->ZeDevice, IsInOrder, Ordinal);
        ASSERT_TRUE(regCmdList != nullptr);

        auto immCmdList = cache.getImmediateCommandList(
            device->ZeDevice, IsInOrder, Ordinal, Mode, Priority);
        ASSERT_TRUE(immCmdList != nullptr);

        ASSERT_EQ(regCmdLists.erase(regCmdList.get()), 1);
        ASSERT_EQ(immCmdLists.erase(immCmdList.get()), 1);
    }
}

TEST_P(CommandListCacheTest, ImmediateCommandListsHaveProperAttributes) {
    v2::command_list_cache_t cache(context->ZeContext);

    uint32_t numQueueGroups = 0;
    ASSERT_EQ(zeDeviceGetCommandQueueGroupProperties(device->ZeDevice,
                                                     &numQueueGroups, nullptr),
              ZE_RESULT_SUCCESS);

    if (numQueueGroups == 0) {
        GTEST_SKIP();
    }

    std::vector<ze_command_queue_group_properties_t> QueueGroupProperties(
        numQueueGroups);
    ASSERT_EQ(
        zeDeviceGetCommandQueueGroupProperties(
            device->ZeDevice, &numQueueGroups, QueueGroupProperties.data()),
        ZE_RESULT_SUCCESS);

    bool IsInOrder = false;

    ze_command_queue_mode_t Mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    ze_command_queue_priority_t Priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

    for (uint32_t Ordinal = 0; Ordinal < numQueueGroups; Ordinal++) {
        // verify list creation with specific indexes
        for (uint32_t Index = 0;
             Index < QueueGroupProperties[Ordinal].numQueues; Index++) {
            auto CommandList = cache.getImmediateCommandList(
                device->ZeDevice, IsInOrder, Ordinal, Mode, Priority, Index);

            ze_device_handle_t ZeDevice;
            ASSERT_EQ(
                zeCommandListGetDeviceHandle(CommandList.get(), &ZeDevice),
                ZE_RESULT_SUCCESS);
            ASSERT_EQ(ZeDevice, device->ZeDevice);

            uint32_t ActualOrdinal;
            ASSERT_EQ(
                zeCommandListGetOrdinal(CommandList.get(), &ActualOrdinal),
                ZE_RESULT_SUCCESS);
            ASSERT_EQ(ActualOrdinal, Ordinal);

            uint32_t ActualIndex;
            ASSERT_EQ(
                zeCommandListImmediateGetIndex(CommandList.get(), &ActualIndex),
                ZE_RESULT_SUCCESS);
            ASSERT_EQ(ActualIndex, Index);

            // store the list back to the cache
            cache.addImmediateCommandList(std::move(CommandList),
                                          device->ZeDevice, IsInOrder, Ordinal,
                                          Mode, Priority, Index);
        }

        // verify list creation without an index
        auto CommandList = cache.getImmediateCommandList(
            device->ZeDevice, IsInOrder, Ordinal, Mode, Priority, std::nullopt);

        ze_device_handle_t ZeDevice;
        ASSERT_EQ(zeCommandListGetDeviceHandle(CommandList.get(), &ZeDevice),
                  ZE_RESULT_SUCCESS);
        ASSERT_EQ(ZeDevice, device->ZeDevice);

        uint32_t ActualOrdinal;
        ASSERT_EQ(zeCommandListGetOrdinal(CommandList.get(), &ActualOrdinal),
                  ZE_RESULT_SUCCESS);
        ASSERT_EQ(ActualOrdinal, Ordinal);

        uint32_t ActualIndex;
        ASSERT_EQ(
            zeCommandListImmediateGetIndex(CommandList.get(), &ActualIndex),
            ZE_RESULT_SUCCESS);
        ASSERT_EQ(ActualIndex, 0);
    }
}
