//==----------- GetNativeOpenCL.cpp ---  interop unit test only for opencl -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS
#define __SYCL_INTERNAL_API

#include <detail/context_impl.hpp>
#include <sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>

#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <memory>

using namespace sycl;

int TestCounter = 0;
int DeviceRetainCounter = 0;

static ur_result_t redefinedContextRetain(void *) {
  ++TestCounter;
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedQueueRetain(void *) {
  ++TestCounter;
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedDeviceRetain(void *) {
  ++TestCounter;
  ++DeviceRetainCounter;
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedProgramRetain(void *) {
  ++TestCounter;
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedEventRetain(void *) {
  ++TestCounter;
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedMemRetain(void *) {
  ++TestCounter;
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedMemBufferCreate(void *) { return UR_RESULT_SUCCESS; }

ur_result_t redefinedEventGetInfo(void *pParams) {
  auto params = *static_cast<ur_event_get_info_params_t *>(pParams);
  EXPECT_EQ(*params.ppropName, UR_EVENT_INFO_COMMAND_EXECUTION_STATUS)
      << "Unexpected event info requested";
  // Report half of events as complete
  static int Counter = 0;
  auto *Result = reinterpret_cast<ur_event_status_t *>(*params.ppPropValue);
  *Result =
      (++Counter % 2 == 0) ? UR_EVENT_STATUS_COMPLETE : UR_EVENT_STATUS_RUNNING;
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedEnqueueUSMFill(void *pParams) {
  auto params = *static_cast<ur_enqueue_usm_fill_params_t *>(pParams);
  **params.pphEvent = reinterpret_cast<ur_event_handle_t>(new int{});
  return UR_RESULT_SUCCESS;
}

TEST(GetNative, GetNativeHandle) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  mock::getCallbacks().set_before_callback("urEventGetInfo",
                                           &redefinedEventGetInfo);
  mock::getCallbacks().set_before_callback("urContextRetain",
                                           &redefinedContextRetain);
  mock::getCallbacks().set_before_callback("urQueueRetain",
                                           &redefinedQueueRetain);
  mock::getCallbacks().set_before_callback("urDeviceRetain",
                                           &redefinedDeviceRetain);
  mock::getCallbacks().set_before_callback("urProgramRetain",
                                           &redefinedProgramRetain);
  mock::getCallbacks().set_before_callback("urEventRetain",
                                           &redefinedEventRetain);
  mock::getCallbacks().set_before_callback("urMemRetain", &redefinedMemRetain);
  mock::getCallbacks().set_before_callback("urMemBufferCreate",
                                           &redefinedMemBufferCreate);
  mock::getCallbacks().set_before_callback("urEnqueueUSMFill",
                                           &redefinedEnqueueUSMFill);

  context Context(Plt);
  queue Queue(Context, default_selector_v);

  auto Device = Queue.get_device();

  unsigned char *HostAlloc = (unsigned char *)malloc_host(1, Context);
  auto Event = Queue.memset(HostAlloc, 42, 1);

  int Data[1] = {0};
  sycl::buffer<int, 1> Buffer(&Data[0], sycl::range<1>(1));
  Queue.submit([&](sycl::handler &cgh) {
    auto Acc = Buffer.get_access<sycl::access::mode::read_write>(cgh);
    constexpr size_t KS = sizeof(decltype(Acc));
    cgh.single_task<TestKernel<KS>>([=]() { (void)Acc; });
  });

  get_native<backend::opencl>(Context);
  get_native<backend::opencl>(Queue);
  get_native<backend::opencl>(Device);
  get_native<backend::opencl>(Event);
  get_native<backend::opencl>(Buffer);

  // Depending on global caches state, urDeviceRetain is called either once or
  // twice, so there'll be 6 or 7 calls.
  ASSERT_EQ(TestCounter, 6 + DeviceRetainCounter - 1)
      << "Not all the retain methods were called";
}
