// REQUIRES: level_zero
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The batch-size dimension is specific to the Level Zero implementation.
// Keep this coverage separate from the backend-independent ordering test.

#define DISCARD_EVENTS_L0_BATCH_TEST
#include "urEnqueueUSMOperationsOrderingIOQ.cpp"

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urEnqueueUSMOperationsOrderingIOQL0Test,
    testing::Values(QueueParameter(UR_QUEUE_FLAG_SUBMISSION_BATCHED, 0),
                    QueueParameter(UR_QUEUE_FLAG_SUBMISSION_BATCHED, 1),
                    QueueParameter(UR_QUEUE_FLAG_SUBMISSION_BATCHED, 2),
                    QueueParameter(UR_QUEUE_FLAG_SUBMISSION_BATCHED, 3),
                    QueueParameter(UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE, 0)),
    PrintQueueParam);

TEST_P(urEnqueueUSMOperationsOrderingIOQL0Test,
       InOrderDiscardEventsOrderingL0BatchSizes) {
  if (!isLevelZeroBackend()) {
    GTEST_SKIP() << "Level Zero batch-size coverage is only applicable to the "
                    "Level Zero backend.";
  }

  runOrderingTestForSupportedUSMTypes();
}