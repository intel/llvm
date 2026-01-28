// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_CONFORMANCE_GRAPH_FIXTURES_H
#define UR_CONFORMANCE_GRAPH_FIXTURES_H

#include "uur/fixtures.h"
#include "uur/known_failure.h"
#include "uur/raii.h"

namespace uur {

struct urGraphSupportedExpTest : uur::urQueueTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());

    UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{}, uur::NativeCPU{},
                         uur::OpenCL{}, uur::LevelZero{});
  }
};

struct urGraphSupportedExpMultiQueueTest : uur::urMultiQueueTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urMultiQueueTest::SetUp());

    ur_bool_t graph_supported = false;
    ur_result_t result = urDeviceGetInfo(
        device, UR_DEVICE_INFO_GRAPH_RECORD_AND_REPLAY_SUPPORT_EXP,
        sizeof(graph_supported), &graph_supported, nullptr);
    if (result == UR_RESULT_SUCCESS && !graph_supported) {
      GTEST_SKIP();
    }

    UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{}, uur::NativeCPU{},
                         uur::OpenCL{}, uur::LevelZero{});
  }
};

struct urGraphExpTest : urGraphSupportedExpTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urGraphSupportedExpTest::SetUp());

    ASSERT_SUCCESS(urGraphCreateExp(context, &graph));
  }

  void TearDown() override {
    if (graph) {
      ASSERT_SUCCESS(urGraphDestroyExp(graph));
    }

    UUR_RETURN_ON_FATAL_FAILURE(urGraphSupportedExpTest::TearDown());
  }

  ur_exp_graph_handle_t graph = nullptr;
};

struct urGraphPopulatedExpTest : urGraphExpTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urGraphExpTest::SetUp());

    ur_device_usm_access_capability_flags_t deviceUSMSupport = 0;
    ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, deviceUSMSupport));
    if (!deviceUSMSupport) {
      GTEST_SKIP() << "Device USM is not supported";
    }
    uur::generateMemFillPattern(pattern);

    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                    allocationSize, &deviceMem));

    ASSERT_SUCCESS(urQueueBeginCaptureIntoGraphExp(queue, graph));

    ASSERT_SUCCESS(urEnqueueUSMFill(queue, deviceMem, patternSize,
                                    pattern.data(), allocationSize, 0, nullptr,
                                    nullptr));

    ur_exp_graph_handle_t sameGraph = nullptr;
    ASSERT_SUCCESS(urQueueEndGraphCaptureExp(queue, &sameGraph));
    ASSERT_EQ(graph, sameGraph);
  }

  void TearDown() override {
    if (deviceMem) {
      ASSERT_SUCCESS(urUSMFree(context, deviceMem));
      resetData();
    }

    UUR_RETURN_ON_FATAL_FAILURE(urGraphExpTest::TearDown());
  }

  void verifyData(const bool shouldMatch) {
    ASSERT_SUCCESS(urEnqueueUSMMemcpy(queue, true, hostMem.data(), deviceMem,
                                      allocationSize, 0, nullptr, nullptr));

    int cmpResult = memcmp(hostMem.data(), pattern.data(), pattern.size());
    ASSERT_EQ(cmpResult == 0, shouldMatch);
  }

  void resetData() {
    const uint8_t zero = 0;
    ASSERT_SUCCESS(urEnqueueUSMFill(queue, deviceMem, sizeof(zero), &zero,
                                    allocationSize, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
  }

  const size_t allocationSize = 256;
  void *deviceMem{nullptr};
  std::vector<uint8_t> hostMem = std::vector<uint8_t>(allocationSize);
  const size_t patternSize = 64;
  std::vector<uint8_t> pattern = std::vector<uint8_t>(patternSize);
};

struct urGraphExecutableExpTest : urGraphPopulatedExpTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urGraphPopulatedExpTest::SetUp());

    ASSERT_SUCCESS(urGraphInstantiateGraphExp(graph, &exGraph));
  }

  void TearDown() override {
    if (exGraph) {
      ASSERT_SUCCESS(urGraphExecutableGraphDestroyExp(exGraph));
    }

    UUR_RETURN_ON_FATAL_FAILURE(urGraphPopulatedExpTest::TearDown());
  }

  ur_exp_executable_graph_handle_t exGraph = nullptr;
};

} // namespace uur

#endif // UR_CONFORMANCE_GRAPH_FIXTURES_H
