//==- KernelBuildOptions.cpp - Kernel build options processing unit test --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#define SYCL2020_DISABLE_DEPRECATION_WARNINGS
#ifndef __SYCL_INTERNAL_API
#define __SYCL_INTERNAL_API
#endif

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

class BuildOptsTestKernel;

static std::string BuildOpts;
namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<BuildOptsTestKernel> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "BuildOptsTestKernel"; }
  static constexpr bool isESIMD() { return true; }
};

} // namespace detail
} // namespace _V1
} // namespace sycl

static ur_result_t redefinedProgramBuild(void *pParams) {
  auto params = *static_cast<ur_program_build_exp_params_t *>(pParams);
  if (*params.ppOptions)
    BuildOpts = *params.ppOptions;
  else
    BuildOpts = "";
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedProgramCompile(void *pParams) {
  auto params = *static_cast<ur_program_compile_exp_params_t *>(pParams);
  if (*params.ppOptions)
    BuildOpts = *params.ppOptions;
  else
    BuildOpts = "";
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedProgramLink(void *pParams) {
  auto params = *static_cast<ur_program_link_exp_params_t *>(pParams);
  if (*params.ppOptions)
    BuildOpts = *params.ppOptions;
  else
    BuildOpts = "";
  return UR_RESULT_SUCCESS;
}

static void setupCommonMockAPIs(sycl::unittest::UrMock<> &Mock) {
  using namespace sycl::detail;
  mock::getCallbacks().set_before_callback("urProgramCompileExp",
                                           &redefinedProgramCompile);
  mock::getCallbacks().set_before_callback("urProgramLinkExp",
                                           &redefinedProgramLink);
  mock::getCallbacks().set_before_callback("urProgramBuildExp",
                                           &redefinedProgramBuild);
}

static sycl::unittest::MockDeviceImage generateDefaultImage() {
  using namespace sycl::unittest;

  MockPropertySet PropSet;
  addESIMDFlag(PropSet);
  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  std::vector<MockOffloadEntry> Entries =
      makeEmptyKernels({"BuildOptsTestKernel"});

  MockDeviceImage Img{SYCL_DEVICE_BINARY_TYPE_SPIRV,       // Format
                      __SYCL_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
                      "-compile-img",                      // Compile options
                      "-link-img",                         // Link options
                      std::move(Bin),
                      std::move(Entries),
                      std::move(PropSet)};

  return Img;
}

sycl::unittest::MockDeviceImage Img = generateDefaultImage();
sycl::unittest::MockDeviceImageArray<1> ImgArray{&Img};

TEST(KernelBuildOptions, KernelBundleBasic) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  setupCommonMockAPIs(Mock);

  const sycl::device Dev = Plt.get_devices()[0];

  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();

  auto KernelID = sycl::get_kernel_id<BuildOptsTestKernel>();
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {KernelID});
  try {
    // unsupported property
    auto ExecBundle = sycl::build(KernelBundle, sycl::property::no_init{});
    FAIL();
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "The property list contains property unsupported "
                           "for the current object");
  }
  auto ExecBundle = sycl::build(KernelBundle);
  EXPECT_EQ(BuildOpts,
            "-compile-img -vc-codegen -disable-finalizer-msg -link-img");
  try {
    auto ObjBundle = sycl::compile(KernelBundle, KernelBundle.get_devices(),
                                   sycl::property::no_init{});
    FAIL();
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "The property list contains property unsupported "
                           "for the current object");
  }
  auto ObjBundle = sycl::compile(KernelBundle, KernelBundle.get_devices());
  EXPECT_EQ(BuildOpts, "-compile-img -vc-codegen -disable-finalizer-msg");

  try {
    auto LinkBundle = sycl::link(ObjBundle, ObjBundle.get_devices(),
                                 sycl::property::no_init{});
    FAIL();
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "The property list contains property unsupported "
                           "for the current object");
  }
  auto LinkBundle = sycl::link(ObjBundle, ObjBundle.get_devices());
  EXPECT_EQ(BuildOpts, "-link-img");
}

TEST(KernelBuildOptions, ESIMDParallelForBasic) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  setupCommonMockAPIs(Mock);

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::queue Queue{Dev};

  Queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<BuildOptsTestKernel>(
        sycl::range{1024}, [=](sycl::id<1>) /* SYCL_ESIMD_KERNEL */ {});
  });

  EXPECT_EQ(BuildOpts,
            "-compile-img -vc-codegen -disable-finalizer-msg -link-img");
}
