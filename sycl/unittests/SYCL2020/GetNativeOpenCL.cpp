//==----------- GetNativeOpenCL.cpp ---  interop unit test only for opencl -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS
#define __SYCL_INTERNAL_API

#include <CL/sycl.hpp>
#include <CL/sycl/backend/opencl.hpp>
#include <detail/context_impl.hpp>

#include <helpers/sycl_test.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <memory>

using namespace cl::sycl;

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

static pi_result redefinedProgramCreateWithSource(pi_context context,
                                                  pi_uint32 count,
                                                  const char **strings,
                                                  const size_t *lengths,
                                                  pi_program *ret_program) {
  *ret_program = reinterpret_cast<pi_program>(new size_t{1});
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

static pi_result redefinedUSMEnqueueMemset(pi_queue, void *, pi_int32, size_t,
                                           pi_uint32, const pi_event *,
                                           pi_event *event) {
  *event = reinterpret_cast<pi_event>(new int{});
  return PI_SUCCESS;
}

SYCL_TEST(GetNative, GetNativeHandle) {
  platform Plt;
  for (auto &Cur : platform::get_platforms()) {
    if (Cur.get_backend() == backend::opencl) {
      Plt = Cur;
      break;
    }
  }
  if (Plt.get_backend() != backend::opencl) {
    std::cout << "Test is created for opencl only" << std::endl;
    return;
  }
  if (Plt.is_host()) {
    std::cout << "Not run on host - no PI events created in that case"
              << std::endl;
    return;
  }
  TestCounter = 0;

  using namespace sycl::unittest;
  redefine<detail::PiApiKind::piEventGetInfo>(redefinedEventGetInfo);
  redefine<detail::PiApiKind::piContextRetain>(redefinedContextRetain);
  redefine<detail::PiApiKind::piQueueRetain>(redefinedQueueRetain);
  redefine<detail::PiApiKind::piDeviceRetain>(redefinedDeviceRetain);
  redefine<detail::PiApiKind::piProgramRetain>(redefinedProgramRetain);
  redefine<detail::PiApiKind::piEventRetain>(redefinedEventRetain);
  redefine<detail::PiApiKind::piextUSMEnqueueMemset>(redefinedUSMEnqueueMemset);
  redefine<detail::PiApiKind::piclProgramCreateWithSource>(
      redefinedProgramCreateWithSource);

  default_selector Selector;
  context Context(Plt.get_devices()[0]);
  queue Queue(Context, Selector);

  program Program{Context};
  Program.build_with_source("");

  auto Device = Queue.get_device();

  unsigned char *HostAlloc = (unsigned char *)malloc_host(1, Context);
  auto Event = Queue.memset(HostAlloc, 42, 1);

  get_native<backend::opencl>(Context);
  get_native<backend::opencl>(Queue);
  get_native<backend::opencl>(Program);
  get_native<backend::opencl>(Device);
  get_native<backend::opencl>(Event);

  // Depending on global caches state, piDeviceRetain is called either once or
  // twice, so there'll be 5 or 6 calls.
  ASSERT_EQ(TestCounter, 5 + DeviceRetainCounter - 1)
      << "Not all the retain methods were called";
}
