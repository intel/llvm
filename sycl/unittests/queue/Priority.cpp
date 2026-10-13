//==---------- Priority.cpp --- check queue priority handling in RT --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/sycl.hpp>

#include <optional>

namespace prop = sycl::ext::oneapi::property::queue;

namespace {

bool PrioritySupported = false;
constexpr int32_t HighestPriority = -4;
constexpr int32_t LowestPriority = 3;

ur_queue_flags_t CreatedFlags = 0;
std::optional<int32_t> CreatedPriority;
bool CreatedWithIndex = false;

template <typename T> ur_result_t returnValue(void *pParams, T Value) {
  auto *Params = static_cast<ur_device_get_info_params_t *>(pParams);
  if (*Params->ppPropValue)
    *static_cast<T *>(*Params->ppPropValue) = Value;
  if (*Params->ppPropSizeRet)
    **Params->ppPropSizeRet = sizeof(T);
  return UR_RESULT_SUCCESS;
}

ur_result_t urDeviceGetInfoAfter(void *pParams) {
  auto *Params = static_cast<ur_device_get_info_params_t *>(pParams);
  switch (*Params->ppropName) {
  case UR_DEVICE_INFO_QUEUE_PRIORITY_SUPPORT_EXP:
    return returnValue<ur_bool_t>(pParams, PrioritySupported);
  case UR_DEVICE_INFO_QUEUE_PRIORITY_HIGHEST_EXP:
    return PrioritySupported ? returnValue<int32_t>(pParams, HighestPriority)
                             : UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  case UR_DEVICE_INFO_QUEUE_PRIORITY_LOWEST_EXP:
    return PrioritySupported ? returnValue<int32_t>(pParams, LowestPriority)
                             : UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  case UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES:
    return returnValue<int32_t>(pParams, 8);
  default:
    return UR_RESULT_SUCCESS;
  }
}

ur_result_t urQueueCreateBefore(void *pParams) {
  auto *Params = static_cast<ur_queue_create_params_t *>(pParams);
  const ur_queue_properties_t *Props = *Params->ppProperties;
  CreatedFlags = Props ? Props->flags : 0;
  CreatedPriority.reset();
  CreatedWithIndex = false;
  for (auto *Next =
           Props ? static_cast<const ur_base_properties_t *>(Props->pNext)
                 : nullptr;
       Next; Next = static_cast<const ur_base_properties_t *>(Next->pNext)) {
    if (Next->stype == UR_STRUCTURE_TYPE_EXP_QUEUE_PRIORITY_PROPERTIES)
      CreatedPriority =
          reinterpret_cast<const ur_exp_queue_priority_properties_t *>(Next)
              ->priority;
    else if (Next->stype == UR_STRUCTURE_TYPE_QUEUE_INDEX_PROPERTIES)
      CreatedWithIndex = true;
  }
  return UR_RESULT_SUCCESS;
}

class QueuePriorityTest : public ::testing::TestWithParam<bool> {
protected:
  void SetUp() override {
    PrioritySupported = GetParam();
    mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                            &urDeviceGetInfoAfter);
    mock::getCallbacks().set_before_callback("urQueueCreate",
                                             &urQueueCreateBefore);
  }

  void expectCreatedWith(std::optional<int32_t> Priority,
                         ur_queue_flags_t PriorityFlags) {
    EXPECT_EQ(CreatedPriority, Priority);
    EXPECT_EQ(CreatedFlags &
                  (UR_QUEUE_FLAG_PRIORITY_LOW | UR_QUEUE_FLAG_PRIORITY_HIGH),
              PriorityFlags);
  }

  sycl::unittest::UrMock<> Mock;
};

} // namespace

TEST_P(QueuePriorityTest, PriorityRange) {
  sycl::device Dev;
  auto Range = Dev.get_info<sycl::ext::oneapi::info::device::priority_range>();
  if (PrioritySupported) {
    EXPECT_EQ(Range[0], LowestPriority);
    EXPECT_EQ(Range[1], HighestPriority);
  } else {
    EXPECT_EQ(Range[0], 0);
    EXPECT_EQ(Range[1], 0);
  }
}

TEST_P(QueuePriorityTest, NumericPriority) {
  sycl::queue Q{prop::priority{-2}};
  ASSERT_TRUE(Q.has_property<prop::priority>());
  EXPECT_EQ(Q.get_property<prop::priority>().get_priority(), -2);
  // Without priority levels the range is {0, 0}, so -2 clamps to default.
  expectCreatedWith(
      PrioritySupported ? std::optional<int32_t>{-2} : std::nullopt, 0);
}

TEST_P(QueuePriorityTest, NumericPriorityIsClamped) {
  {
    sycl::queue Q{prop::priority{-100}};
    expectCreatedWith(PrioritySupported
                          ? std::optional<int32_t>{HighestPriority}
                          : std::nullopt,
                      0);
  }
  {
    sycl::queue Q{prop::priority{100}};
    expectCreatedWith(PrioritySupported ? std::optional<int32_t>{LowestPriority}
                                        : std::nullopt,
                      0);
  }
}

TEST_P(QueuePriorityTest, DefaultPriorityIsNotPassed) {
  {
    sycl::queue Q{prop::priority{0}};
    expectCreatedWith(std::nullopt, 0);
  }
  {
    sycl::queue Q{prop::priority_normal{}};
    expectCreatedWith(std::nullopt, 0);
  }
}

TEST_P(QueuePriorityTest, PriorityHigh) {
  sycl::queue Q{prop::priority_high{}};
  if (PrioritySupported)
    expectCreatedWith(HighestPriority, 0);
  else
    expectCreatedWith(std::nullopt, UR_QUEUE_FLAG_PRIORITY_HIGH);
}

TEST_P(QueuePriorityTest, PriorityLowUsesFlag) {
  sycl::queue Q{prop::priority_low{}};
  expectCreatedWith(std::nullopt, UR_QUEUE_FLAG_PRIORITY_LOW);
}

TEST_P(QueuePriorityTest, WithComputeIndex) {
  sycl::queue Q{{prop::priority{-1},
                 sycl::ext::intel::property::queue::compute_index{1}}};
  EXPECT_TRUE(CreatedWithIndex);
  expectCreatedWith(
      PrioritySupported ? std::optional<int32_t>{-1} : std::nullopt, 0);
}

TEST_P(QueuePriorityTest, ConflictingPriorities) {
  auto ExpectInvalid = [](sycl::property_list Props) {
    try {
      sycl::queue Q{Props};
      FAIL() << "Expected an exception";
    } catch (sycl::exception &E) {
      EXPECT_EQ(E.code(), sycl::errc::invalid);
    }
  };
  ExpectInvalid({prop::priority{1}, prop::priority_normal{}});
  ExpectInvalid({prop::priority{1}, prop::priority_low{}});
  ExpectInvalid({prop::priority{1}, prop::priority_high{}});
  ExpectInvalid({prop::priority_low{}, prop::priority_high{}});
}

INSTANTIATE_TEST_SUITE_P(QueuePriority, QueuePriorityTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool> &Info) {
                           return Info.param ? "LevelsSupported"
                                             : "LevelsUnsupported";
                         });
