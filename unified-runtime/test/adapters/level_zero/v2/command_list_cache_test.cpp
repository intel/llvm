// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "command_list_cache.hpp"
#include "context.hpp"

#include "level_zero/common.hpp"
#include "level_zero/device.hpp"

#include "uur/fixtures.h"
#include "uur/raii.h"

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

struct CommandListCacheTest : public uur::urContextTest {
  void SetUp() override {
    // Initialize Level Zero driver is required if this test is linked
    // statically with Level Zero loader, the driver will not be init otherwise.
    zeInit(ZE_INIT_FLAG_GPU_ONLY);
    urContextTest::SetUp();
  }
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE(CommandListCacheTest);

TEST_P(CommandListCacheTest, CanStoreAndRetriveImmediateAndRegularCmdLists) {
  v2::command_list_cache_t cache(context->getZeHandle(), false);

  bool IsInOrder = false;
  uint32_t Ordinal = 0;

  ze_command_queue_mode_t Mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
  ze_command_queue_priority_t Priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

  static constexpr int numListsPerType = 3;
  std::vector<v2::raii::command_list_unique_handle> regCmdListOwners;
  std::vector<v2::raii::command_list_unique_handle> immCmdListOwners;

  std::unordered_set<ze_command_list_handle_t> regCmdLists;
  std::unordered_set<ze_command_list_handle_t> immCmdLists;

  // get command lists from the cache
  for (int i = 0; i < numListsPerType; ++i) {
    regCmdListOwners.emplace_back(cache.getRegularCommandList(
        device->ZeDevice, IsInOrder, Ordinal, true));
    auto [it, _] = regCmdLists.emplace(regCmdListOwners.back().get());
    ASSERT_TRUE(*it != nullptr);

    immCmdListOwners.emplace_back(cache.getImmediateCommandList(
        device->ZeDevice, IsInOrder, Ordinal, true, Mode, Priority));
    std::tie(it, _) = immCmdLists.emplace(immCmdListOwners.back().get());
    ASSERT_TRUE(*it != nullptr);
  }

  // store them back into the cache
  regCmdListOwners.clear();
  immCmdListOwners.clear();

  // verify we get back the same command lists
  for (int i = 0; i < numListsPerType; ++i) {
    auto regCmdList =
        cache.getRegularCommandList(device->ZeDevice, IsInOrder, Ordinal, true);
    ASSERT_TRUE(regCmdList != nullptr);

    auto immCmdList = cache.getImmediateCommandList(
        device->ZeDevice, IsInOrder, Ordinal, true, Mode, Priority);
    ASSERT_TRUE(immCmdList != nullptr);

    ASSERT_EQ(regCmdLists.erase(regCmdList.get()), 1);
    ASSERT_EQ(immCmdLists.erase(immCmdList.get()), 1);

    // release the command list manually so they are not added back to the cache
    zeCommandListDestroy(regCmdList.release());
    zeCommandListDestroy(immCmdList.release());
  }
}

TEST_P(CommandListCacheTest, ImmediateCommandListsHaveProperAttributes) {
  v2::command_list_cache_t cache(context->getZeHandle(), false);

  uint32_t numQueueGroups = 0;
  ASSERT_EQ(zeDeviceGetCommandQueueGroupProperties(device->ZeDevice,
                                                   &numQueueGroups, nullptr),
            ZE_RESULT_SUCCESS);

  if (numQueueGroups == 0) {
    GTEST_SKIP();
  }

  std::vector<ze_command_queue_group_properties_t> QueueGroupProperties(
      numQueueGroups);
  ASSERT_EQ(zeDeviceGetCommandQueueGroupProperties(
                device->ZeDevice, &numQueueGroups, QueueGroupProperties.data()),
            ZE_RESULT_SUCCESS);

  bool IsInOrder = false;

  ze_command_queue_mode_t Mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
  ze_command_queue_priority_t Priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

  for (uint32_t Ordinal = 0; Ordinal < numQueueGroups; Ordinal++) {
    // verify list creation with specific indexes
    for (uint32_t Index = 0; Index < QueueGroupProperties[Ordinal].numQueues;
         Index++) {
      auto CommandList = cache.getImmediateCommandList(
          device->ZeDevice, IsInOrder, Ordinal, true, Mode, Priority, Index);

      ze_device_handle_t ZeDevice;
      auto Ret = zeCommandListGetDeviceHandle(CommandList.get(), &ZeDevice);
      if (Ret == ZE_RESULT_SUCCESS) {
        ASSERT_EQ(ZeDevice, device->ZeDevice);
      } else {
        ASSERT_EQ(Ret, ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
      }

      uint32_t ActualOrdinal;
      Ret = zeCommandListGetOrdinal(CommandList.get(), &ActualOrdinal);
      if (Ret == ZE_RESULT_SUCCESS) {
        ASSERT_EQ(ActualOrdinal, Ordinal);
      } else {
        ASSERT_EQ(Ret, ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
      }

      uint32_t ActualIndex;
      Ret = zeCommandListImmediateGetIndex(CommandList.get(), &ActualIndex);
      if (Ret == ZE_RESULT_SUCCESS) {
        ASSERT_EQ(ActualIndex, Index);
      } else {
        ASSERT_EQ(Ret, ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
      }
    }

    // verify list creation without an index
    auto CommandList =
        cache.getImmediateCommandList(device->ZeDevice, IsInOrder, Ordinal,
                                      true, Mode, Priority, std::nullopt);

    ze_device_handle_t ZeDevice;
    auto Ret = zeCommandListGetDeviceHandle(CommandList.get(), &ZeDevice);
    if (Ret == ZE_RESULT_SUCCESS) {
      ASSERT_EQ(ZeDevice, device->ZeDevice);
    } else {
      ASSERT_EQ(Ret, ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
    }

    uint32_t ActualOrdinal;
    Ret = zeCommandListGetOrdinal(CommandList.get(), &ActualOrdinal);
    if (Ret == ZE_RESULT_SUCCESS) {
      ASSERT_EQ(ActualOrdinal, Ordinal);
    } else {
      ASSERT_EQ(Ret, ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
    }

    uint32_t ActualIndex;
    Ret = zeCommandListImmediateGetIndex(CommandList.get(), &ActualIndex);
    if (Ret == ZE_RESULT_SUCCESS) {
      ASSERT_EQ(ActualIndex, 0);
    } else {
      ASSERT_EQ(Ret, ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
    }
  }
}

TEST_P(CommandListCacheTest, CommandListsAreReusedByQueues) {
  static constexpr int NumQueuesPerType = 5;
  size_t NumUniqueQueueTypes = 0;

  for (int I = 0; I < NumQueuesPerType; I++) {
    NumUniqueQueueTypes = 0;

    { // Queues scope
      std::vector<uur::raii::Queue> Queues;
      for (auto Priority : std::vector<uint32_t>{
               UR_QUEUE_FLAG_PRIORITY_LOW, UR_QUEUE_FLAG_PRIORITY_HIGH, 0}) {
        for (auto Index :
             std::vector<std::optional<int32_t>>{std::nullopt, 0}) {
          NumUniqueQueueTypes++;

          ur_queue_properties_t QueueProps{UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
                                           nullptr, 0};
          QueueProps.flags |= UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE;
          if (Priority) {
            QueueProps.flags |= Priority;
          }

          ur_queue_index_properties_t IndexProps{
              UR_STRUCTURE_TYPE_QUEUE_INDEX_PROPERTIES, nullptr, 0};
          if (Index) {
            IndexProps.computeIndex = *Index;
            QueueProps.pNext = &IndexProps;
          }

          uur::raii::Queue Queue;
          ASSERT_EQ(urQueueCreate(context, device, &QueueProps, Queue.ptr()),
                    UR_RESULT_SUCCESS);

          Queues.emplace_back(Queue);
        }
      }

      ASSERT_EQ(context->getCommandListCache().getNumImmediateCommandLists(),
                0);
      ASSERT_EQ(context->getCommandListCache().getNumRegularCommandLists(), 0);
    } // Queues scope

    ASSERT_EQ(context->getCommandListCache().getNumImmediateCommandLists(),
              NumUniqueQueueTypes);
    ASSERT_EQ(context->getCommandListCache().getNumRegularCommandLists(), 0);
  }
}

TEST_P(CommandListCacheTest, CommandListsCacheIsThreadSafe) {
  static constexpr int NumThreads = 10;
  static constexpr int NumIters = 10;

  std::vector<std::thread> Threads;
  for (int I = 0; I < NumThreads; I++) {
    Threads.emplace_back([I, this]() {
      for (int J = 0; J < NumIters; J++) {
        ur_queue_properties_t QueueProps{UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
                                         nullptr, 0};
        QueueProps.flags |= UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE;
        if (I < NumThreads / 2) {
          QueueProps.flags |= UR_QUEUE_FLAG_PRIORITY_LOW;
        } else {
          QueueProps.flags |= UR_QUEUE_FLAG_PRIORITY_HIGH;
        }

        uur::raii::Queue Queue;
        ASSERT_EQ(urQueueCreate(context, device, &QueueProps, Queue.ptr()),
                  UR_RESULT_SUCCESS);

        ASSERT_LE(context->getCommandListCache().getNumImmediateCommandLists(),
                  NumThreads);
      }
    });
  }

  for (auto &Thread : Threads) {
    Thread.join();
  }

  ASSERT_LE(context->getCommandListCache().getNumImmediateCommandLists(),
            NumThreads);
}
