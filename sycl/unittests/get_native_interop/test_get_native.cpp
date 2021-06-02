//==----------- test_get_native.cpp --- get_native interop unit test only for
// opencl
//-------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <CL/sycl.hpp>
#include <CL/sycl/backend/opencl.hpp>
#include <detail/context_impl.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>

#include <iostream>
#include <memory>

using namespace cl::sycl;

int TestCounter;

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
  return PI_SUCCESS;
}

static pi_result
redefinedProgramBuild(pi_program program, pi_uint32 num_devices,
                      const pi_device *device_list, const char *options,
                      void (*pfn_notify)(pi_program program, void *user_data),
                      void *user_data) {
  return PI_SUCCESS;
}

pi_result redefinedEventsWait(pi_uint32 num_events,
                              const pi_event *event_list) {
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

pi_result redefinedEventRelease(pi_event event) { return PI_SUCCESS; }

TEST(GetNativeTest, GetNativeHandle) {
  platform Plt{default_selector()};
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

  unittest::PiMock Mock{Plt};
  Mock.redefine<detail::PiApiKind::piclProgramCreateWithSource>(
      redefinedProgramCreateWithSource);
  Mock.redefine<detail::PiApiKind::piProgramBuild>(redefinedProgramBuild);

  Mock.redefine<detail::PiApiKind::piEventsWait>(redefinedEventsWait);
  Mock.redefine<detail::PiApiKind::piEventGetInfo>(redefinedEventGetInfo);
  Mock.redefine<detail::PiApiKind::piEventRelease>(redefinedEventRelease);

  Mock.redefine<detail::PiApiKind::piContextRetain>(redefinedContextRetain);
  Mock.redefine<detail::PiApiKind::piQueueRetain>(redefinedQueueRetain);
  Mock.redefine<detail::PiApiKind::piDeviceRetain>(redefinedDeviceRetain);
  Mock.redefine<detail::PiApiKind::piProgramRetain>(redefinedProgramRetain);
  Mock.redefine<detail::PiApiKind::piEventRetain>(redefinedEventRetain);

  default_selector Selector;
  context Context(Plt);
  queue Queue(Context, Selector);

  program Program{Context};
  Program.build_with_source("");

  auto Device = Queue.get_device();

  unsigned char *HostAlloc = (unsigned char *)malloc_host(1, Context);
  auto Event = Queue.memset(HostAlloc, 42, 1);

  auto c_g = get_native<backend::opencl>(Context);
  auto q_g = get_native<backend::opencl>(Queue);
  auto p_g = get_native<backend::opencl>(Program);
  auto d_g = get_native<backend::opencl>(Device);
  auto e_g = get_native<backend::opencl>(Event);

  // When creating a context, the piDeviceRetain is called so here is the 6
  // retain calls
  ASSERT_EQ(TestCounter, 6) << "Not all the retain methods was called";
}
