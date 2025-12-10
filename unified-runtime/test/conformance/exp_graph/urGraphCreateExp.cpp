// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urGraphCreateExpTest = uur::urGraphSupportedExpTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urGraphCreateExpTest);

TEST_P(urGraphCreateExpTest, Success) {
  ur_exp_graph_handle_t graph = nullptr;
  ASSERT_SUCCESS(urGraphCreateExp(context, &graph));
  ASSERT_SUCCESS(urGraphDestroyExp(graph));
}

TEST_P(urGraphCreateExpTest, InvalidNullHandleContext) {
  ur_exp_graph_handle_t graph = nullptr;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urGraphCreateExp(nullptr, &graph));
}

TEST_P(urGraphCreateExpTest, InvalidNullPtrGraph) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urGraphCreateExp(context, nullptr));
}
