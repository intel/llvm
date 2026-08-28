// REQUIRES: level_zero
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The batch-size dimension is specific to the Level Zero implementation.
// Keep this coverage separate from the backend-independent ordering test.
//
// UR_L0_BATCH_SIZE is only read once, into a process-wide static, when the
// adapter is first loaded, so it cannot be varied via a gtest parameter:
// re-invoke the whole binary once per batch size instead.
// RUN: env UR_L0_BATCH_SIZE=0 %maybe-v1 ./usm_ops_ordering_l0-test
// RUN: env UR_L0_BATCH_SIZE=1 %maybe-v1 ./usm_ops_ordering_l0-test
// RUN: env UR_L0_BATCH_SIZE=2 %maybe-v1 ./usm_ops_ordering_l0-test
// RUN: env UR_L0_BATCH_SIZE=3 %maybe-v1 ./usm_ops_ordering_l0-test

#include "../../conformance/enqueue/urEnqueueUSMOperationsOrderingIOQ.hpp"

struct urEnqueueUSMOperationsOrderingIOQL0Test
    : urEnqueueUSMOperationsOrderingIOQTestBase {};

UUR_DEVICE_TEST_SUITE_WITH_PARAM(
    urEnqueueUSMOperationsOrderingIOQL0Test,
    testing::Values(UR_QUEUE_FLAG_SUBMISSION_BATCHED,
                    UR_QUEUE_FLAG_SUBMISSION_IMMEDIATE),
    PrintQueueParam);

TEST_P(urEnqueueUSMOperationsOrderingIOQL0Test,
       InOrderDiscardEventsOrderingL0BatchSizes) {
  if (!isLevelZeroBackend()) {
    GTEST_SKIP() << "Level Zero batch-size coverage is only applicable to the "
                    "Level Zero backend.";
  }

  runOrderingTestForSupportedUSMTypes();
}
