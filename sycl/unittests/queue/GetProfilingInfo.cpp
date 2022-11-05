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

#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <detail/context_impl.hpp>

class InfoTestKernel;

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
template <> struct KernelInfo<InfoTestKernel> {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return "InfoTestKernel"; }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
  static constexpr int64_t getKernelSize() { return 1; }
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
template <typename T> sycl::unittest::PiImage generateTestImage() {
  using namespace sycl::unittest;

  PiPropertySet PropSet;

  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  PiArray<PiOffloadEntry> Entries = makeEmptyKernels({"InfoTestKernel"});

  PiImage Img{PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
              "",                                     // Compile options
              "",                                     // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}

static pi_result
redefinedPiEventGetProfilingInfo(pi_event event, pi_profiling_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t *param_value_size_ret) {
  return PI_SUCCESS;
}

TEST(GetProfilingInfo, normal_pass_without_exception) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefineBefore<sycl::detail::PiApiKind::piEventGetProfilingInfo>(
      redefinedPiEventGetProfilingInfo);
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::context Ctx{Dev};
  static sycl::unittest::PiImage DevImage_1 =
      generateTestImage<InfoTestKernel>();

  static sycl::unittest::PiImageArray<1> DevImageArray = {&DevImage_1};
  auto KernelID_1 = sycl::get_kernel_id<InfoTestKernel>();
  sycl::queue Queue{
      Ctx, Dev, sycl::property_list{sycl::property::queue::enable_profiling{}}};
  auto KernelBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
      Ctx, {Dev}, {KernelID_1});

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
  static sycl::unittest::PiImage DevImage_1 =
      generateTestImage<InfoTestKernel>();

  static sycl::unittest::PiImageArray<1> DevImageArray = {&DevImage_1};
  auto KernelID_1 = sycl::get_kernel_id<InfoTestKernel>();
  sycl::queue Queue{Ctx, Dev};
  auto KernelBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
      Ctx, {Dev}, {KernelID_1});
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
  static sycl::unittest::PiImage DevImage_1 =
      generateTestImage<InfoTestKernel>();

  static sycl::unittest::PiImageArray<1> DevImageArray = {&DevImage_1};
  auto KernelID_1 = sycl::get_kernel_id<InfoTestKernel>();
  const int globalWIs{512};
  sycl::event event;
  {
    sycl::queue Queue{
        Ctx, Dev,
        sycl::property_list{sycl::property::queue::enable_profiling{}}};
    auto KernelBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
        Ctx, {Dev}, {KernelID_1});
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
  static sycl::unittest::PiImage DevImage_1 =
      generateTestImage<InfoTestKernel>();

  static sycl::unittest::PiImageArray<1> DevImageArray = {&DevImage_1};
  auto KernelID_1 = sycl::get_kernel_id<InfoTestKernel>();
  const int globalWIs{512};
  sycl::event event;
  {
    sycl::queue Queue{Ctx, Dev};
    auto KernelBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
        Ctx, {Dev}, {KernelID_1});
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
