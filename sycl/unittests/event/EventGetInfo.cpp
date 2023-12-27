//==------------------ EventGetInfo.cpp --- event unit tests ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

pi_result redefinedEventGetInfoAfter(pi_event event, pi_event_info param_name,
                                     size_t param_value_size, void *param_value,
                                     size_t *param_value_size_ret) {
  EXPECT_EQ(param_name, PI_EVENT_INFO_COMMAND_EXECUTION_STATUS)
      << "Unexpected event info requested";
  static std::unordered_map<pi_event, bool> events;

  auto *Result = reinterpret_cast<pi_event_status *>(param_value);
  if (events.find(event) == events.end()) {
    *Result = PI_EVENT_QUEUED;
    events.insert({event, true});
  }

  return PI_SUCCESS;
}

void preparePiMock(unittest::PiMock &Mock) {
  Mock.redefineAfter<detail::PiApiKind::piEventGetInfo>(
      redefinedEventGetInfoAfter);
}

// Check that events that are in queued status are handled properly.
TEST(EventGetInfo, EventQueuedStatus) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  preparePiMock(Mock);

  context Ctx{Plt.get_devices()[0]};
  queue Q{Ctx, default_selector()};

  auto DeviceAlloc = sycl::malloc_device<char>(1, Q);

  auto event = Q.memset(DeviceAlloc, 42, 1);
  auto info = event.get_info<sycl::info::event::command_execution_status>();

  ASSERT_TRUE(info == sycl::info::event_command_status::complete ||
              info == sycl::info::event_command_status::running ||
              info == sycl::info::event_command_status::submitted);

  sycl::free(DeviceAlloc, Q);
}
