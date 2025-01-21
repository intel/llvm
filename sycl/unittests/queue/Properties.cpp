//==-------- Properties.cpp --- check properties handling in RT --- --------==//
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

template <typename PropertyType> void DatalessQueuePropertyCheck() {
  try {
    sycl::queue Queue{PropertyType{}};
    ASSERT_TRUE(Queue.has_property<PropertyType>());
    Queue.get_property<PropertyType>();
    // no explicit checks, we expect no exception to be thrown
  } catch (...) {
    FAIL();
  }
}

TEST(QueueProperties, ValidDatalessProperties) {
  sycl::unittest::UrMock<> Mock;
  DatalessQueuePropertyCheck<sycl::property::queue::in_order>();
  DatalessQueuePropertyCheck<sycl::property::queue::enable_profiling>();
  DatalessQueuePropertyCheck<
      sycl::ext::oneapi::property::queue::discard_events>();
  DatalessQueuePropertyCheck<
      sycl::ext::oneapi::property::queue::priority_normal>();
  DatalessQueuePropertyCheck<
      sycl::ext::oneapi::property::queue::priority_low>();
  DatalessQueuePropertyCheck<
      sycl::ext::oneapi::property::queue::priority_high>();
  DatalessQueuePropertyCheck<
      sycl::ext::intel::property::queue::no_immediate_command_list>();
  DatalessQueuePropertyCheck<
      sycl::ext::intel::property::queue::immediate_command_list>();
  DatalessQueuePropertyCheck<
      sycl::ext::oneapi::cuda::property::queue::use_default_stream>();
}

inline ur_result_t urDeviceGetInfoRedefined(void *pParams) {
  auto params = reinterpret_cast<ur_device_get_info_params_t *>(pParams);
  switch (*params->ppropName) {
  case UR_DEVICE_INFO_MAX_COMPUTE_QUEUE_INDICES: {
    if (*params->ppPropValue)
      *static_cast<int32_t *>(*params->ppPropValue) = 8;
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(int32_t);
    return UR_RESULT_SUCCESS;
  }
  default:
    return UR_RESULT_SUCCESS;
  }
}

TEST(QueueProperties, ValidPropertyComputeIndex) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &urDeviceGetInfoRedefined);
  try {
    sycl::queue Queue{sycl::ext::intel::property::queue::compute_index{1}};
    ASSERT_TRUE(
        Queue.has_property<sycl::ext::intel::property::queue::compute_index>());
    EXPECT_EQ(
        Queue.get_property<sycl::ext::intel::property::queue::compute_index>()
            .get_index(),
        1);
    // no explicit checks, we expect no exception to be thrown
  } catch (...) {
    FAIL();
  }
}

TEST(QueueProperties, SetUnsupportedParam) {
  sycl::unittest::UrMock<> Mock;
  try {
    sycl::queue Queue{sycl::property::image::use_host_ptr{}};
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "The property list contains property unsupported "
                           "for the current object");
    return;
  }

  FAIL() << "Test must exit in exception handler. Exception is not thrown.";
}
