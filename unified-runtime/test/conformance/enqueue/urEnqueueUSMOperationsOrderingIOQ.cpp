// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "urEnqueueUSMOperationsOrderingIOQ.hpp"

struct urEnqueueUSMOperationsOrderingIOQTest
    : urEnqueueUSMOperationsOrderingIOQTestBase {};

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urEnqueueUSMOperationsOrderingIOQTest,
    testing::Values(QueueParameter(UR_QUEUE_FLAG_SUBMISSION_BATCHED),
                    QueueParameter(UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE)),
    PrintQueueParam);

TEST_P(urEnqueueUSMOperationsOrderingIOQTest, InOrderDiscardEventsOrdering) {
  runOrderingTestForSupportedUSMTypes();
}
