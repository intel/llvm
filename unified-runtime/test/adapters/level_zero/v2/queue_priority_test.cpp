// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %with-v2 ./queue_priority-test
// REQUIRES: v2

#include "level_zero/common/device.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <utility>

using namespace ur::level_zero;

namespace {

constexpr int32_t Min = std::numeric_limits<int32_t>::min();
constexpr int32_t Max = std::numeric_limits<int32_t>::max();

void expectPriority(const ze_queue_priority &Actual,
                    ze_command_queue_priority_t Class,
                    std::optional<int32_t> Level) {
  EXPECT_EQ(Actual.Class, Class);
  EXPECT_EQ(Actual.Level, Level);
}

} // namespace

TEST(QueuePriorityRangeTest, WithoutDriverLevelsUsesPriorityClasses) {
  // No zeDeviceGetPriorityLevels, a single level, or a range without 0.
  for (auto DriverLevels :
       {std::optional<std::pair<int32_t, int32_t>>{},
        std::optional{std::pair{0, 0}}, std::optional{std::pair{2, 1}}}) {
    const auto Range = makeQueuePriorityRange(DriverLevels);
    EXPECT_EQ(Range.Highest, -1);
    EXPECT_EQ(Range.Lowest, 1);
    EXPECT_FALSE(Range.HasLevels);
  }
}

TEST(QueuePriorityRangeTest, WithDriverLevelsAddsLowestLevel) {
  const auto Range = makeQueuePriorityRange(std::pair{0, -3});
  EXPECT_EQ(Range.Highest, -3);
  EXPECT_EQ(Range.Lowest, 1);
  EXPECT_TRUE(Range.HasLevels);
}

TEST(QueuePriorityTest, WithoutDriverLevelsUsesPriorityClasses) {
  const auto Range = makeQueuePriorityRange(std::nullopt);

  expectPriority(getZeQueuePriority(Range, 0, std::nullopt),
                 ZE_COMMAND_QUEUE_PRIORITY_NORMAL, std::nullopt);
  expectPriority(
      getZeQueuePriority(Range, UR_QUEUE_FLAG_PRIORITY_LOW, std::nullopt),
      ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW, std::nullopt);
  expectPriority(
      getZeQueuePriority(Range, UR_QUEUE_FLAG_PRIORITY_HIGH, std::nullopt),
      ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH, std::nullopt);

  for (int32_t Priority : {1, Max})
    expectPriority(getZeQueuePriority(Range, 0, Priority),
                   ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW, std::nullopt);
  expectPriority(getZeQueuePriority(Range, 0, 0),
                 ZE_COMMAND_QUEUE_PRIORITY_NORMAL, std::nullopt);
  for (int32_t Priority : {-1, Min})
    expectPriority(getZeQueuePriority(Range, 0, Priority),
                   ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH, std::nullopt);
}

TEST(QueuePriorityTest, WithDriverLevelsUsesPriorityLevels) {
  const auto Range = makeQueuePriorityRange(std::pair{0, -3});

  expectPriority(getZeQueuePriority(Range, 0, std::nullopt),
                 ZE_COMMAND_QUEUE_PRIORITY_NORMAL, std::nullopt);
  expectPriority(
      getZeQueuePriority(Range, UR_QUEUE_FLAG_PRIORITY_LOW, std::nullopt),
      ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW, std::nullopt);
  expectPriority(
      getZeQueuePriority(Range, UR_QUEUE_FLAG_PRIORITY_HIGH, std::nullopt),
      ZE_COMMAND_QUEUE_PRIORITY_NORMAL, -3);

  for (int32_t Priority : {1, Max})
    expectPriority(getZeQueuePriority(Range, 0, Priority),
                   ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW, std::nullopt);
  for (int32_t Priority : {0, -2})
    expectPriority(getZeQueuePriority(Range, 0, Priority),
                   ZE_COMMAND_QUEUE_PRIORITY_NORMAL, Priority);
  expectPriority(getZeQueuePriority(Range, 0, Min),
                 ZE_COMMAND_QUEUE_PRIORITY_NORMAL, -3);
}

TEST(QueuePriorityTest, HighFlagMatchesHighestLevelWhenHighestIsZero) {
  const auto Range = makeQueuePriorityRange(std::pair{1, 0});
  ASSERT_TRUE(Range.HasLevels);
  ASSERT_EQ(Range.Highest, 0);

  const auto HighFlag =
      getZeQueuePriority(Range, UR_QUEUE_FLAG_PRIORITY_HIGH, std::nullopt);
  const auto HighestLevel = getZeQueuePriority(Range, 0, Range.Highest);
  expectPriority(HighFlag, ZE_COMMAND_QUEUE_PRIORITY_NORMAL, 0);
  expectPriority(HighestLevel, HighFlag.Class, HighFlag.Level);
}
