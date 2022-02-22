//==------------------ GetProfilingInfo.cpp --- unit tests for check if needed
//exception is thrown
// when get_profiling_info() called without queue::enable_profiling property in
// queue property list -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <cstring>
#include <gtest/gtest.h>

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

class InfoTestKernel;

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
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
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
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
  cl::sycl::platform Plt{cl::sycl::default_selector{}};
  if (Plt.is_host()) {
    GTEST_SKIP();
  }
  sycl::unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  Mock.redefine<sycl::detail::PiApiKind::piEventGetProfilingInfo>(
      redefinedPiEventGetProfilingInfo);
  const sycl::device Dev = Plt.get_devices()[0];
  static sycl::unittest::PiImage DevImage_1 =
      generateTestImage<InfoTestKernel>();

  static sycl::unittest::PiImageArray<1> DevImageArray = {&DevImage_1};
  auto KernelID_1 = sycl::get_kernel_id<InfoTestKernel>();
  sycl::queue Queue{
      Dev, sycl::property_list{sycl::property::queue::enable_profiling{}}};
  const sycl::context Ctx = Queue.get_context();
  auto KernelBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
      Ctx, {Dev}, {KernelID_1});

  const int globalWIs{512};
  EXPECT_NO_THROW({
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
  });
}

TEST(GetProfilingInfo, command_exception_check) {
  cl::sycl::platform Plt{cl::sycl::default_selector{}};
  if (Plt.is_host()) {
    GTEST_SKIP();
  }
  sycl::unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  Mock.redefine<sycl::detail::PiApiKind::piEventGetProfilingInfo>(
      redefinedPiEventGetProfilingInfo);

  const sycl::device Dev = Plt.get_devices()[0];
  static sycl::unittest::PiImage DevImage_1 =
      generateTestImage<InfoTestKernel>();

  static sycl::unittest::PiImageArray<1> DevImageArray = {&DevImage_1};
  auto KernelID_1 = sycl::get_kernel_id<InfoTestKernel>();
  {
    sycl::queue Queue{Dev};
    const sycl::context Ctx = Queue.get_context();
    auto KernelBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
        Ctx, {Dev}, {KernelID_1});
    const int globalWIs{512};
    try {
      auto event = Queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<InfoTestKernel>(globalWIs, [=](sycl::id<1> idx) {});
      });
      event.wait();
      auto submit_time = event.get_profiling_info<
          sycl::info::event_profiling::command_submit>();
      (void)submit_time;
    } catch (sycl::exception const &e) {
      std::cerr << e.what() << std::endl;
      EXPECT_STREQ(e.what(), "get_profiling_info() can't be used without set "
                             "'enable_profiling' queue property");
    }
  }
  {
    sycl::queue Queue{Dev};
    const sycl::context Ctx = Queue.get_context();
    auto KernelBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
        Ctx, {Dev}, {KernelID_1});
    const int globalWIs{512};
    try {
      auto event = Queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<InfoTestKernel>(globalWIs, [=](sycl::id<1> idx) {});
      });
      event.wait();
      auto start_time =
          event
              .get_profiling_info<sycl::info::event_profiling::command_start>();
      (void)start_time;
    } catch (sycl::exception const &e) {
      std::cerr << e.what() << std::endl;
      EXPECT_STREQ(e.what(), "get_profiling_info() can't be used without set "
                             "'enable_profiling' queue property");
    }
  }
  {
    sycl::queue Queue{Dev};
    const sycl::context Ctx = Queue.get_context();
    auto KernelBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
        Ctx, {Dev}, {KernelID_1});
    const int globalWIs{512};
    try {
      auto event = Queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<InfoTestKernel>(globalWIs, [=](sycl::id<1> idx) {});
      });
      event.wait();
      auto end_time =
          event.get_profiling_info<sycl::info::event_profiling::command_end>();
      (void)end_time;
    } catch (sycl::exception const &e) {
      EXPECT_STREQ(e.what(), "get_profiling_info() can't be used without set "
                             "'enable_profiling' queue property");
    }
  }
}
