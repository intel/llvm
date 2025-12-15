// Copyright (C) 2025 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.h"

using urGraphNonEmptyExpTest = uur::urGraphPopulatedExpTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urGraphNonEmptyExpTest);

TEST_P(urGraphNonEmptyExpTest, SuccessFalse) {
  bool isEmpty = false;
  ASSERT_SUCCESS(urGraphIsEmptyExp(graph, &isEmpty));
  ASSERT_TRUE(isEmpty);
}

using urGraphEmptyExpTest = uur::urGraphExpTest;

UUR_INSTANTIATE_DEVICE_TEST_SUITE(urGraphEmptyExpTest);

TEST_P(urGraphEmptyExpTest, SuccessTrue) {
  bool isEmpty = false;
  ASSERT_SUCCESS(urGraphIsEmptyExp(graph, &isEmpty));
  ASSERT_FALSE(isEmpty);
}

TEST_P(urGraphEmptyExpTest, InvalidNullHandleQueue) {
  bool isEmpty = false;
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                   urGraphIsEmptyExp(nullptr, &isEmpty));
}

TEST_P(urGraphEmptyExpTest, InvalidNullPtrResult) {
  ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                   urGraphIsEmptyExp(graph, nullptr));
}
