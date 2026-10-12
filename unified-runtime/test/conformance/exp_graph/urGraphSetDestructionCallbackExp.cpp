// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

struct urGraphSetDestructionCallbackExpTest : uur::urGraphSupportedExpTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urGraphSupportedExpTest::SetUp());
    std::tuple<size_t, size_t, size_t> minL0DriverVersion = {1, 15, 37561};
    SKIP_IF_DRIVER_TOO_OLD("Level-Zero", minL0DriverVersion, platform, device);
    ASSERT_SUCCESS(urGraphCreateExp(context, &graph));
  }

  void TearDown() override {
    if (graph) {
      urGraphDestroyExp(graph);
    }
    UUR_RETURN_ON_FATAL_FAILURE(urGraphSupportedExpTest::TearDown());
  }

  ur_exp_graph_handle_t graph = nullptr;
};

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urGraphSetDestructionCallbackExpTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urGraphSetDestructionCallbackExpTest, SuccessFreeMemory) {
  int *data = static_cast<int *>(malloc(sizeof(int)));
  ASSERT_NE(data, nullptr);
  *data = 42;

  ur_exp_graph_destruction_callback_t callback = +[](void *pUserData) {
    void **pCastedData = static_cast<void **>(pUserData);
    free(*pCastedData);
    *pCastedData = nullptr;
  };
  ASSERT_SUCCESS(urGraphSetDestructionCallbackExp(graph, callback,
                                                  static_cast<void *>(&data)));

  ASSERT_NE(data, nullptr);
  ASSERT_EQ(*data, 42);
  ASSERT_SUCCESS(urGraphDestroyExp(graph));
  graph = nullptr;
  ASSERT_EQ(data, nullptr);
}

TEST_P(urGraphSetDestructionCallbackExpTest, SuccessMultipleCallbacks) {
  bool firstInvoked = false;
  bool secondInvoked = false;

  ur_exp_graph_destruction_callback_t callback =
      +[](void *pUserData) { *static_cast<bool *>(pUserData) = true; };
  ASSERT_SUCCESS(
      urGraphSetDestructionCallbackExp(graph, callback, &firstInvoked));
  ASSERT_SUCCESS(
      urGraphSetDestructionCallbackExp(graph, callback, &secondInvoked));

  ASSERT_FALSE(firstInvoked);
  ASSERT_FALSE(secondInvoked);
  ASSERT_SUCCESS(urGraphDestroyExp(graph));
  graph = nullptr;
  ASSERT_TRUE(firstInvoked);
  ASSERT_TRUE(secondInvoked);
}

TEST_P(urGraphSetDestructionCallbackExpTest, InvalidNullHandleGraph) {
  ASSERT_EQ_RESULT(
      UR_RESULT_ERROR_INVALID_NULL_HANDLE,
      urGraphSetDestructionCallbackExp(nullptr, [](void *) {}, nullptr));
}

TEST_P(urGraphSetDestructionCallbackExpTest, InvalidNullPointerCallback) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urGraphSetDestructionCallbackExp(graph, nullptr, nullptr));
}
