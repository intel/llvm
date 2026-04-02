// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urGraphInstantiateGraphExpTest = uur::urGraphExpTest;

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urGraphInstantiateGraphExpTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urGraphInstantiateGraphExpTest, InvalidEmptyGraph) {
  ur_exp_executable_graph_handle_t exGraph = nullptr;
  ASSERT_EQ(urGraphInstantiateGraphExp(graph, &exGraph),
            UR_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(urGraphInstantiateGraphExpTest, InvalidNullHandleGraph) {
  ur_exp_executable_graph_handle_t exGraph = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urGraphInstantiateGraphExp(nullptr, &exGraph));
}

TEST_P(urGraphInstantiateGraphExpTest, InvalidNullPtrExGraph) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urGraphInstantiateGraphExp(graph, nullptr));
}

using urGraphInstantiatePopulatedGraphExpTest = uur::urGraphPopulatedExpTest;

UUR_DEVICE_TEST_SUITE_WITH_QUEUE_TYPES(
    urGraphInstantiatePopulatedGraphExpTest,
    ::testing::Values(0 /* In-Order */,
                      UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE));

TEST_P(urGraphInstantiatePopulatedGraphExpTest, SuccessMultipleInstantiations) {
  const size_t numInstances = 5;
  std::vector<ur_exp_executable_graph_handle_t> exGraphs(numInstances, nullptr);

  for (size_t i = 0; i < numInstances; ++i) {
    ASSERT_SUCCESS(urGraphInstantiateGraphExp(graph, &exGraphs[i]));
    ASSERT_NE(exGraphs[i], nullptr);
  }

  for (size_t i = 0; i < numInstances; ++i) {
    for (size_t j = i + 1; j < numInstances; ++j) {
      ASSERT_NE(exGraphs[i], exGraphs[j]);
    }
  }

  for (size_t i = 0; i < numInstances; ++i) {
    ASSERT_SUCCESS(urGraphExecutableGraphDestroyExp(exGraphs[i]));
  }
}
