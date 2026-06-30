// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

#include <vector>

struct urGraphGetIdExpTest : uur::urGraphExpTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urGraphExpTest::SetUp());
    // Unique graph id support was added in this driver version.
    std::tuple<size_t, size_t, size_t> minL0DriverVersion = {1, 15, 38921};
    SKIP_IF_DRIVER_TOO_OLD("Level-Zero", minL0DriverVersion, platform, device);
  }
};

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urGraphGetIdExpTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urGraphGetIdExpTest, SuccessMonotonicallyIncreasing) {
  uint64_t currentGraphId = 0;
  ASSERT_SUCCESS(urGraphGetIdExp(graph, &currentGraphId));

  // Ensure the IDs increase monotonically when graphs are kept alive
  std::vector<ur_exp_graph_handle_t> graphs;
  for (int i = 0; i < 5; i++) {
    ur_exp_graph_handle_t nextGraph = nullptr;
    ASSERT_SUCCESS(urGraphCreateExp(context, &nextGraph));
    graphs.push_back(nextGraph);

    uint64_t nextGraphId = 0;
    ASSERT_SUCCESS(urGraphGetIdExp(nextGraph, &nextGraphId));
    ASSERT_GT(nextGraphId, currentGraphId);
    currentGraphId = nextGraphId;
  }

  // Ensure the IDs increase monotonically even when graphs are destroyed
  for (int i = 0; i < 5; i++) {
    ur_exp_graph_handle_t nextGraph = nullptr;
    ASSERT_SUCCESS(urGraphCreateExp(context, &nextGraph));

    uint64_t nextGraphId = 0;
    ASSERT_SUCCESS(urGraphGetIdExp(nextGraph, &nextGraphId));
    ASSERT_GT(nextGraphId, currentGraphId);
    currentGraphId = nextGraphId;

    ASSERT_SUCCESS(urGraphDestroyExp(nextGraph));
  }

  for (ur_exp_graph_handle_t currentGraph : graphs) {
    ASSERT_SUCCESS(urGraphDestroyExp(currentGraph));
  }
}

TEST_P(urGraphGetIdExpTest, SuccessUniqueAcrossContexts) {
  uint64_t firstGraphId = 0;
  ASSERT_SUCCESS(urGraphGetIdExp(graph, &firstGraphId));

  ur_context_handle_t secondContext = nullptr;
  ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, &secondContext));

  ur_exp_graph_handle_t secondGraph = nullptr;
  ASSERT_SUCCESS(urGraphCreateExp(secondContext, &secondGraph));

  uint64_t secondGraphId = 0;
  ASSERT_SUCCESS(urGraphGetIdExp(secondGraph, &secondGraphId));
  ASSERT_GT(secondGraphId, firstGraphId);

  ASSERT_SUCCESS(urGraphDestroyExp(secondGraph));
  ASSERT_SUCCESS(urContextRelease(secondContext));
}

TEST_P(urGraphGetIdExpTest, InvalidNullHandleGraph) {
  uint64_t graphId = 0;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urGraphGetIdExp(nullptr, &graphId));
}

TEST_P(urGraphGetIdExpTest, InvalidNullPtrGraphId) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urGraphGetIdExp(graph, nullptr));
}
