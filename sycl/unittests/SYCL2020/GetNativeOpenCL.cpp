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

#include <helpers/PiMock.hpp>
#include <helpers/TestKernel.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <memory>

using namespace sycl;

int TestCounter = 0;
int DeviceRetainCounter = 0;

static pi_result redefinedContextRetain(pi_context c) {
  ++TestCounter;
  return PI_SUCCESS;
}

static pi_result redefinedQueueRetain(pi_queue c) {
  ++TestCounter;
  return PI_SUCCESS;
}

static pi_result redefinedDeviceRetain(pi_device c) {
  ++TestCounter;
  ++DeviceRetainCounter;
  return PI_SUCCESS;
}

static pi_result redefinedProgramRetain(pi_program c) {
  ++TestCounter;
  return PI_SUCCESS;
}

static pi_result redefinedEventRetain(pi_event c) {
  ++TestCounter;
  return PI_SUCCESS;
}

static pi_result redefinedMemRetain(pi_mem c) {
  ++TestCounter;
  return PI_SUCCESS;
}

pi_result redefinedMemBufferCreate(pi_context, pi_mem_flags, size_t size,
                                   void *, pi_mem *,
                                   const pi_mem_properties *) {
  return PI_SUCCESS;
}

pi_result redefinedEventGetInfo(pi_event event, pi_event_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
  EXPECT_EQ(param_name, PI_EVENT_INFO_COMMAND_EXECUTION_STATUS)
      << "Unexpected event info requested";
  // Report half of events as complete
  static int Counter = 0;
  auto *Result = reinterpret_cast<pi_event_status *>(param_value);
  *Result = (++Counter % 2 == 0) ? PI_EVENT_COMPLETE : PI_EVENT_RUNNING;
  return PI_SUCCESS;
}

static pi_result redefinedUSMEnqueueMemset(pi_queue, void *, const void *,
                                           size_t, size_t, pi_uint32,
                                           const pi_event *, pi_event *event) {
  *event = reinterpret_cast<pi_event>(new int{});
  return PI_SUCCESS;
}

TEST(GetNative, GetNativeHandle) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();

  Mock.redefineBefore<detail::PiApiKind::piEventGetInfo>(redefinedEventGetInfo);
  Mock.redefineBefore<detail::PiApiKind::piContextRetain>(
      redefinedContextRetain);
  Mock.redefineBefore<detail::PiApiKind::piQueueRetain>(redefinedQueueRetain);
  Mock.redefineBefore<detail::PiApiKind::piDeviceRetain>(redefinedDeviceRetain);
  Mock.redefineBefore<detail::PiApiKind::piProgramRetain>(
      redefinedProgramRetain);
  Mock.redefineBefore<detail::PiApiKind::piEventRetain>(redefinedEventRetain);
  Mock.redefineBefore<detail::PiApiKind::piMemRetain>(redefinedMemRetain);
  Mock.redefineBefore<sycl::detail::PiApiKind::piMemBufferCreate>(
      redefinedMemBufferCreate);
  Mock.redefineBefore<detail::PiApiKind::piextUSMEnqueueFill>(
      redefinedUSMEnqueueMemset);

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

  // Depending on global caches state, piDeviceRetain is called either once or
  // twice, so there'll be 6 or 7 calls.
  ASSERT_EQ(TestCounter, 6 + DeviceRetainCounter - 1)
      << "Not all the retain methods were called";
}
