//==------------------ GetProfilingInfo.cpp --- unit tests for check if needed
// exception is thrown
// when get_profiling_info() called without queue::enable_profiling property in
// queue property list -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

#include <sycl/detail/defines_elementary.hpp>

#include <helpers/MockKernelInfo.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>
#include <helpers/TestKernel.hpp>

#include <detail/context_impl.hpp>

class InfoTestKernel;

MOCK_INTEGRATION_HEADER(InfoTestKernel)

static pi_result
redefinedPiEventGetProfilingInfo(pi_event event, pi_profiling_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

static pi_result redefinedPiDevicesGet(pi_platform platform,
                                       pi_device_type device_type,
                                       pi_uint32 num_entries,
                                       pi_device *devices,
                                       pi_uint32 *num_devices) {
  // Host/Device timer syncronization isn't done all the time (cached), so we
  // need brand new device for some of the testcases.
  static std::intptr_t device_id = 10;
  if (num_devices)
    *num_devices = 1;

  if (devices && num_entries > 0)
    devices[0] = reinterpret_cast<pi_device>(++device_id);

  return PI_SUCCESS;
}

TEST(GetProfilingInfo, normal_pass_without_exception) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefineBefore<sycl::detail::PiApiKind::piEventGetProfilingInfo>(
      redefinedPiEventGetProfilingInfo);
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  static auto DevImage =
      sycl::unittest::generateDefaultImage({"InfoTestKernel"});

  static sycl::unittest::PiImageArray<1> DevImageArray = {&DevImage};
  auto KernelID = sycl::get_kernel_id<InfoTestKernel>();
  sycl::queue Queue{
      Ctx, Dev, sycl::property_list{sycl::property::queue::enable_profiling{}}};
  auto KernelBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
      Ctx, {Dev}, {KernelID});

  const int globalWIs{512};
  try {
    auto event = Queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<InfoTestKernel>(globalWIs, [=](sycl::id<1> idx) {});
    });
    event.wait();
    auto submit_time =
        event.get_profiling_info<sycl::info::event_profiling::command_submit>();
    auto start_time =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end_time =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();
    (void)submit_time;
    (void)start_time;
    (void)end_time;
  } catch (sycl::exception const &e) {
    std::cerr << e.what() << std::endl;
    FAIL();
  }
}

TEST(GetProfilingInfo, command_exception_check) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefineBefore<sycl::detail::PiApiKind::piEventGetProfilingInfo>(
      redefinedPiEventGetProfilingInfo);
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  static auto DevImage =
      sycl::unittest::generateDefaultImage({"InfoTestKernel"});
  static sycl::unittest::PiImageArray<1> DevImageArray = {&DevImage};
  auto KernelID = sycl::get_kernel_id<InfoTestKernel>();
  sycl::queue Queue{Ctx, Dev};
  auto KernelBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
      Ctx, {Dev}, {KernelID});
  const int globalWIs{512};
  {
    try {
      auto event = Queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<InfoTestKernel>(globalWIs, [=](sycl::id<1> idx) {});
      });
      event.wait();
      auto submit_time = event.get_profiling_info<
          sycl::info::event_profiling::command_submit>();
      (void)submit_time;
      FAIL();
    } catch (sycl::exception &e) {
      EXPECT_STREQ(
          e.what(),
          "Profiling information is unavailable as the queue associated with "
          "the event does not have the 'enable_profiling' property.");
    }
  }
  {
    try {
      auto event = Queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<InfoTestKernel>(globalWIs, [=](sycl::id<1> idx) {});
      });
      event.wait();
      auto start_time =
          event
              .get_profiling_info<sycl::info::event_profiling::command_start>();
      (void)start_time;
      FAIL();
    } catch (sycl::exception const &e) {
      std::cerr << e.what() << std::endl;
      EXPECT_STREQ(
          e.what(),
          "Profiling information is unavailable as the queue associated with "
          "the event does not have the 'enable_profiling' property.");
    }
  }
  {
    try {
      auto event = Queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<InfoTestKernel>(globalWIs, [=](sycl::id<1> idx) {});
      });
      event.wait();
      auto end_time =
          event.get_profiling_info<sycl::info::event_profiling::command_end>();
      (void)end_time;
      FAIL();
    } catch (sycl::exception const &e) {
      EXPECT_STREQ(
          e.what(),
          "Profiling information is unavailable as the queue associated with "
          "the event does not have the 'enable_profiling' property.");
    }
  }
}

TEST(GetProfilingInfo, exception_check_no_queue) {
  sycl::event E;
  try {
    auto info =
        E.get_profiling_info<sycl::info::event_profiling::command_submit>();
    (void)info;
    FAIL();
  } catch (sycl::exception const &e) {
    EXPECT_STREQ(e.what(), "Profiling information is unavailable as the event "
                           "has no associated queue.");
  }
  try {
    auto info =
        E.get_profiling_info<sycl::info::event_profiling::command_start>();
    (void)info;
    FAIL();
  } catch (sycl::exception const &e) {
    EXPECT_STREQ(e.what(), "Profiling information is unavailable as the event "
                           "has no associated queue.");
  }
  try {
    auto info =
        E.get_profiling_info<sycl::info::event_profiling::command_end>();
    (void)info;
    FAIL();
  } catch (sycl::exception const &e) {
    EXPECT_STREQ(e.what(), "Profiling information is unavailable as the event "
                           "has no associated queue.");
  }
}

TEST(GetProfilingInfo, check_if_now_dead_queue_property_set) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefineBefore<sycl::detail::PiApiKind::piEventGetProfilingInfo>(
      redefinedPiEventGetProfilingInfo);
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  static auto DevImage =
      sycl::unittest::generateDefaultImage({"InfoTestKernel"});
  static sycl::unittest::PiImageArray<1> DevImageArray = {&DevImage};
  auto KernelID = sycl::get_kernel_id<InfoTestKernel>();
  const int globalWIs{512};
  sycl::event event;
  {
    sycl::queue Queue{
        Ctx, Dev,
        sycl::property_list{sycl::property::queue::enable_profiling{}}};
    auto KernelBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
        Ctx, {Dev}, {KernelID});
    event = Queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<InfoTestKernel>(globalWIs, [=](sycl::id<1> idx) {});
    });
    event.wait();
  }
  try {
    auto submit_time =
        event.get_profiling_info<sycl::info::event_profiling::command_submit>();
    auto start_time =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end_time =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();
    (void)submit_time;
    (void)start_time;
    (void)end_time;
  } catch (sycl::exception &e) {
    std::cerr << e.what() << std::endl;
    FAIL();
  }
}

TEST(GetProfilingInfo, check_if_now_dead_queue_property_not_set) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefineBefore<sycl::detail::PiApiKind::piEventGetProfilingInfo>(
      redefinedPiEventGetProfilingInfo);
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  static auto DevImage =
      sycl::unittest::generateDefaultImage({"InfoTestKernel"});

  static sycl::unittest::PiImageArray<1> DevImageArray = {&DevImage};
  auto KernelID = sycl::get_kernel_id<InfoTestKernel>();
  const int globalWIs{512};
  sycl::event event;
  {
    sycl::queue Queue{Ctx, Dev};
    auto KernelBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
        Ctx, {Dev}, {KernelID});
    event = Queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<InfoTestKernel>(globalWIs, [=](sycl::id<1> idx) {});
    });
    event.wait();
  }
  {
    try {
      auto submit_time = event.get_profiling_info<
          sycl::info::event_profiling::command_submit>();
      (void)submit_time;
      FAIL();
    } catch (sycl::exception &e) {
      EXPECT_STREQ(
          e.what(),
          "Profiling information is unavailable as the queue associated with "
          "the event does not have the 'enable_profiling' property.");
    }
  }
  {
    try {
      auto start_time =
          event
              .get_profiling_info<sycl::info::event_profiling::command_start>();
      (void)start_time;
      FAIL();
    } catch (sycl::exception &e) {
      EXPECT_STREQ(
          e.what(),
          "Profiling information is unavailable as the queue associated with "
          "the event does not have the 'enable_profiling' property.");
    }
  }
  {
    try {
      auto end_time =
          event.get_profiling_info<sycl::info::event_profiling::command_end>();
      (void)end_time;
      FAIL();
    } catch (sycl::exception &e) {
      EXPECT_STREQ(
          e.what(),
          "Profiling information is unavailable as the queue associated with "
          "the event does not have the 'enable_profiling' property.");
    }
  }
  // The test passes without this, but keep it still, just in case.
  sycl::detail::getSyclObjImpl(Ctx)->getKernelProgramCache().reset();
}

bool DeviceTimerCalled;

pi_result redefinedPiGetDeviceAndHostTimer(pi_device Device,
                                           uint64_t *DeviceTime,
                                           uint64_t *HostTime) {
  DeviceTimerCalled = true;
  return PI_SUCCESS;
}

TEST(GetProfilingInfo,
     check_no_command_submission_time_when_event_profiling_disabled) {
  using namespace sycl;
  unittest::PiMock Mock;
  platform Plt = Mock.getPlatform();
  Mock.redefine<sycl::detail::PiApiKind::piDevicesGet>(redefinedPiDevicesGet);
  Mock.redefine<detail::PiApiKind::piGetDeviceAndHostTimer>(
      redefinedPiGetDeviceAndHostTimer);
  device Dev = Plt.get_devices()[0];
  context Ctx{Dev};
  queue Queue{Ctx, Dev};
  DeviceTimerCalled = false;

  event E = Queue.submit(
      [&](handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); });
  EXPECT_FALSE(DeviceTimerCalled);
}

// Checks to see if command submit time is calculated before queue.submit
// returns. A host accessor is contructed before submitting the command, to
// ensure command submission time is calculated even if command may not be
// enqueued due to overlap in data dependencies between the kernel and host
// accessor
TEST(GetProfilingInfo, check_command_submission_time_with_host_accessor) {
  using namespace sycl;
  unittest::PiMock Mock;
  platform Plt = Mock.getPlatform();
  Mock.redefine<sycl::detail::PiApiKind::piDevicesGet>(redefinedPiDevicesGet);
  Mock.redefine<detail::PiApiKind::piGetDeviceAndHostTimer>(
      redefinedPiGetDeviceAndHostTimer);
  device Dev = Plt.get_devices()[0];
  context Ctx{Dev};
  queue Queue{Ctx, Dev, property::queue::enable_profiling()};
  int data[1024];
  buffer Buf{data, range<1>{1024}};
  DeviceTimerCalled = false;

  accessor host_acc = Buf.get_access<access::mode::read_write>();
  event E = Queue.submit([&](handler &cgh) {
    accessor writeRes{Buf, cgh, read_write};

    cgh.single_task<TestKernel<>>([]() {});
  });

  EXPECT_TRUE(DeviceTimerCalled);
}

pi_result redefinedFailedPiGetDeviceAndHostTimer(pi_device Device,
                                                 uint64_t *DeviceTime,
                                                 uint64_t *HostTime) {
  return PI_ERROR_INVALID_OPERATION;
}

static pi_result redefinedDeviceGetInfoAcc(pi_device device,
                                           pi_device_info param_name,
                                           size_t param_value_size,
                                           void *param_value,
                                           size_t *param_value_size_ret) {
  if (param_name == PI_DEVICE_INFO_TYPE) {
    auto *Result = reinterpret_cast<_pi_device_type *>(param_value);
    *Result = PI_DEVICE_TYPE_ACC;
  }
  return PI_SUCCESS;
}

TEST(GetProfilingInfo, fallback_profiling_PiGetDeviceAndHostTimer_unsupported) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefine<sycl::detail::PiApiKind::piDevicesGet>(redefinedPiDevicesGet);
  Mock.redefineBefore<sycl::detail::PiApiKind::piEventGetProfilingInfo>(
      redefinedPiEventGetProfilingInfo);
  Mock.redefine<sycl::detail::PiApiKind::piGetDeviceAndHostTimer>(
      redefinedFailedPiGetDeviceAndHostTimer);
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoAcc);
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  static auto DevImage =
      sycl::unittest::generateDefaultImage({"InfoTestKernel"});
  static sycl::unittest::PiImageArray<1> DevImageArray = {&DevImage};
  auto KernelID = sycl::get_kernel_id<InfoTestKernel>();
  sycl::queue Queue{
      Ctx, Dev, sycl::property_list{sycl::property::queue::enable_profiling{}}};
  auto KernelBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
      Ctx, {Dev}, {KernelID});

  const int globalWIs{512};
  DeviceTimerCalled = true;
  auto event = Queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<InfoTestKernel>(globalWIs, [=](sycl::id<1> idx) {});
  });
  event.wait();
  auto submit_time =
      event.get_profiling_info<sycl::info::event_profiling::command_submit>();
  auto start_time =
      event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end_time =
      event.get_profiling_info<sycl::info::event_profiling::command_end>();
  assert((submit_time && start_time && end_time) &&
         "Profiling information failed.");
  EXPECT_LT(submit_time, start_time);
  EXPECT_LT(submit_time, end_time);
}

TEST(GetProfilingInfo, fallback_profiling_mock_piEnqueueKernelLaunch) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefine<sycl::detail::PiApiKind::piDevicesGet>(redefinedPiDevicesGet);
  Mock.redefineBefore<sycl::detail::PiApiKind::piEventGetProfilingInfo>(
      redefinedPiEventGetProfilingInfo);
  Mock.redefine<sycl::detail::PiApiKind::piGetDeviceAndHostTimer>(
      redefinedFailedPiGetDeviceAndHostTimer);
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoAcc);
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  static auto DevImage =
      sycl::unittest::generateDefaultImage({"InfoTestKernel"});
  static sycl::unittest::PiImageArray<1> DevImageArray = {&DevImage};
  auto KernelID = sycl::get_kernel_id<InfoTestKernel>();
  sycl::queue Queue{
      Ctx, Dev, sycl::property_list{sycl::property::queue::enable_profiling{}}};
  auto KernelBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
      Ctx, {Dev}, {KernelID});

  const int globalWIs{512};
  DeviceTimerCalled = true;
  auto event = Queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<InfoTestKernel>(globalWIs, [=](sycl::id<1> idx) {});
  });
  event.wait();
  auto submit_time =
      event.get_profiling_info<sycl::info::event_profiling::command_submit>();
  auto start_time =
      event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end_time =
      event.get_profiling_info<sycl::info::event_profiling::command_end>();
  assert((submit_time && start_time && end_time) &&
         "Profiling information failed.");
  EXPECT_LT(submit_time, start_time);
  EXPECT_LT(submit_time, end_time);
}
