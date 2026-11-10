// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_CONFORMANCE_GRAPH_FIXTURES_H
#define UR_CONFORMANCE_GRAPH_FIXTURES_H

#include "uur/fixtures.h"
#include "uur/known_failure.h"
#include "uur/raii.h"
#include "uur/utils.h"

#include <tuple>

namespace uur {

struct urGraphSupportedExpTest : uur::urMultiQueueTypeTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urMultiQueueTypeTest::SetUp());

    ur_bool_t graph_supported = false;
    ASSERT_SUCCESS(urDeviceGetInfo(
        device, UR_DEVICE_INFO_GRAPH_RECORD_AND_REPLAY_SUPPORT_EXP,
        sizeof(graph_supported), &graph_supported, nullptr));
    if (!graph_supported) {
      GTEST_SKIP();
    }

    UUR_KNOWN_FAILURE_ON(uur::CUDA{}, uur::HIP{}, uur::NativeCPU{},
                         uur::OpenCL{}, uur::LevelZero{});
  }
};

struct urGraphSupportedExpMultiQueueTest : uur::urMultiQueueMultiTypeTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urMultiQueueMultiTypeTest::SetUp());

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

// Minimum Level Zero driver versions that report the graph capture error codes
// tested in this suite. Older drivers either succeed or return a generic error.
// For example drivers 1.13.35563, 1.14.37435 and 1.15.38646 support graph
// capture but return UNSUPPORTED_FEATURE instead of GRAPH_CAPTURE_UNSUPPORTED
// and INVALID_ARGUMENT instead of COMMAND_LIST_NOT_CAPTURING. 1.17.39395 is the
// oldest driver verified to return the codes gated by the first constant.
inline constexpr std::tuple<size_t, size_t, size_t>
    graphCaptureErrorsMinL0DriverVersion = {1, 17, 39395};
inline constexpr std::tuple<size_t, size_t, size_t>
    graphForkErrorsMinL0DriverVersion = {1, 17, 39428};
inline constexpr std::tuple<size_t, size_t, size_t>
    graphEventWaitErrorsMinL0DriverVersion = {1, 17, 39758};
inline constexpr std::tuple<size_t, size_t, size_t>
    graphInternalEventErrorsMinL0DriverVersion = {1, 17, 40012};

#define UUR_GRAPH_SKIP_IF_L0_DRIVER_OLDER_THAN(version)                        \
  do {                                                                         \
    auto minL0DriverVersion = (version);                                       \
    SKIP_IF_DRIVER_TOO_OLD("Level-Zero", minL0DriverVersion, platform,         \
                           device);                                            \
  } while (0)

// Two queues of the same type, where queue1 is used as the queue on which a
// graph capture begins and queue2 can be forked from it.
struct urGraphMultiQueueCaptureExpTest : urGraphSupportedExpMultiQueueTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urGraphSupportedExpMultiQueueTest::SetUp());

    // Advance the out-of-order queue's command list selection so that forked
    // work doesn't land on its capture command list by chance.
    uur::raii::Event preEvent = nullptr;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue2, 0, nullptr, preEvent.ptr()));
    ASSERT_SUCCESS(urEventWait(1, preEvent.ptr()));
  }

  void TearDown() override {
    endCaptureIfActive(queue1);
    endCaptureIfActive(queue2);
    // Events used by a capture are only released once the capture ended.
    forkEvent = nullptr;
    capturedEvent = nullptr;
    if (queue1) {
      EXPECT_SUCCESS(urQueueFinish(queue1));
    }
    if (queue2) {
      EXPECT_SUCCESS(urQueueFinish(queue2));
    }
    if (graph1) {
      EXPECT_SUCCESS(urGraphDestroyExp(graph1));
    }

    UUR_RETURN_ON_FATAL_FAILURE(urGraphSupportedExpMultiQueueTest::TearDown());
  }

  void endCaptureIfActive(ur_queue_handle_t q) {
    bool isCapturing = false;
    if (q &&
        urQueueIsGraphCaptureEnabledExp(q, &isCapturing) == UR_RESULT_SUCCESS &&
        isCapturing) {
      ur_exp_graph_handle_t g = nullptr;
      // The result is ignored, the capture may end with an error.
      urQueueEndGraphCaptureExp(q, &g);
      if (g && g != graph1) {
        EXPECT_SUCCESS(urGraphDestroyExp(g));
      }
    }
  }

  // Forks queue2 off the capture active on queue1, without joining it back.
  void forkWithoutJoin() {
    ASSERT_SUCCESS(
        urEnqueueEventsWaitWithBarrier(queue1, 0, nullptr, forkEvent.ptr()));
    ASSERT_SUCCESS(urEnqueueEventsWait(queue2, 1, forkEvent.ptr(), nullptr));
  }

  // Captures into graph1 with a fork that is never joined, which leaves
  // graph1 invalid.
  void recordGraphWithUnjoinedForks() {
    ASSERT_SUCCESS(urGraphCreateExp(context, &graph1));
    ASSERT_SUCCESS(urQueueBeginCaptureIntoGraphExp(queue1, graph1));
    UUR_RETURN_ON_FATAL_FAILURE(forkWithoutJoin());

    ur_exp_graph_handle_t capturedGraph = nullptr;
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_GRAPH_UNJOINED_FORKS,
                     urQueueEndGraphCaptureExp(queue1, &capturedGraph));
  }

  uur::raii::Event forkEvent = nullptr;
  uur::raii::Event capturedEvent = nullptr;
  ur_exp_graph_handle_t graph1 = nullptr;
};

} // namespace uur

#endif // UR_CONFORMANCE_GRAPH_FIXTURES_H
