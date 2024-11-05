//==---- passing_link_and_compile_options.cpp --- Test for passing link and
// compile options for online linker and compiler ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/sycl.hpp>

#include <sycl/detail/defines_elementary.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

std::string current_link_options, current_compile_options, current_build_opts;

class EAMTestKernel1;
const char EAMTestKernelName1[] = "LinkCompileTestKernel1";
constexpr unsigned EAMTestKernelNumArgs1 = 4;

class EAMTestKernel2;
const char EAMTestKernelName2[] = "LinkCompileTestKernel2";
constexpr unsigned EAMTestKernelNumArgs2 = 4;

class EAMTestKernel3;
const char EAMTestKernelName3[] = "LinkCompileTestKernel3";
constexpr unsigned EAMTestKernelNumArgs3 = 4;

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<EAMTestKernel1> : public unittest::MockKernelInfoBase {
  static constexpr unsigned getNumParams() { return EAMTestKernelNumArgs1; }
  static constexpr const char *getName() { return EAMTestKernelName1; }
};

template <>
struct KernelInfo<EAMTestKernel2> : public unittest::MockKernelInfoBase {
  static constexpr unsigned getNumParams() { return EAMTestKernelNumArgs2; }
  static constexpr const char *getName() { return EAMTestKernelName2; }
};

template <>
struct KernelInfo<EAMTestKernel3> : public unittest::MockKernelInfoBase {
  static constexpr unsigned getNumParams() { return EAMTestKernelNumArgs3; }
  static constexpr const char *getName() { return EAMTestKernelName3; }
};

} // namespace detail
} // namespace _V1
} // namespace sycl

template <typename T>
static sycl::unittest::MockDeviceImage
generateEAMTestKernelImage(std::string _cmplOptions, std::string _lnkOptions) {
  using namespace sycl::unittest;

  std::vector<unsigned char> KernelEAM1{0b00000101};
  MockProperty EAMKernelPOI =
      makeKernelParamOptInfo(sycl::detail::KernelInfo<T>::getName(),
                             EAMTestKernelNumArgs1, KernelEAM1);
  std::vector<MockProperty> ImgKPOI{std::move(EAMKernelPOI)};

  MockPropertySet PropSet;
  PropSet.insert(__SYCL_PROPERTY_SET_KERNEL_PARAM_OPT_INFO, std::move(ImgKPOI));

  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  std::vector<MockOffloadEntry> Entries =
      makeEmptyKernels({sycl::detail::KernelInfo<T>::getName()});

  MockDeviceImage Img{SYCL_DEVICE_BINARY_TYPE_SPIRV,       // Format
                      __SYCL_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
                      _cmplOptions,                        // Compile options
                      _lnkOptions,                         // Link options
                      std::move(Bin),
                      std::move(Entries),
                      std::move(PropSet)};
  return Img;
}

inline ur_result_t redefinedProgramLink(void *pParams) {
  auto params = *static_cast<ur_program_link_exp_params_t *>(pParams);
  assert(*params.ppOptions != nullptr);
  auto add_link_opts = std::string(*params.ppOptions);
  if (!add_link_opts.empty()) {
    if (!current_link_options.empty())
      current_link_options += " ";
    current_link_options += std::string(*params.ppOptions);
  }
  return UR_RESULT_SUCCESS;
}

inline ur_result_t redefinedProgramCompile(void *pParams) {
  auto params = *static_cast<ur_program_compile_exp_params_t *>(pParams);
  assert(*params.ppOptions != nullptr);
  auto add_compile_opts = std::string(*params.ppOptions);
  if (!add_compile_opts.empty()) {
    if (!current_compile_options.empty())
      current_compile_options += " ";
    current_compile_options += std::string(*params.ppOptions);
  }
  return UR_RESULT_SUCCESS;
}

inline ur_result_t redefinedProgramBuild(void *pParams) {
  auto params = *static_cast<ur_program_build_exp_params_t *>(pParams);
  assert(*params.ppOptions != nullptr);
  current_build_opts = std::string(*params.ppOptions);
  return UR_RESULT_SUCCESS;
}

TEST(Link_Compile_Options, compile_link_Options_Test_empty_options) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_before_callback("urProgramCompileExp",
                                           &redefinedProgramCompile);
  mock::getCallbacks().set_before_callback("urProgramLinkExp",
                                           &redefinedProgramLink);
  const sycl::device Dev = Plt.get_devices()[0];
  current_link_options.clear();
  current_compile_options.clear();
  std::string expected_options = "";
  static sycl::unittest::MockDeviceImage DevImage =
      generateEAMTestKernelImage<EAMTestKernel1>(expected_options,
                                                 expected_options);
  static sycl::unittest::MockDeviceImageArray<1> DevImageArray_{&DevImage};
  auto KernelID_1 = sycl::get_kernel_id<EAMTestKernel1>();
  sycl::queue Queue{Dev};
  const sycl::context Ctx = Queue.get_context();
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {KernelID_1});
  auto BundleObj = sycl::compile(KernelBundle);
  sycl::link(BundleObj);
  EXPECT_EQ(expected_options, current_link_options);
  EXPECT_EQ(expected_options, current_compile_options);
}

TEST(Link_Compile_Options, compile_link_Options_Test_filled_options) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_before_callback("urProgramCompileExp",
                                           &redefinedProgramCompile);
  mock::getCallbacks().set_before_callback("urProgramLinkExp",
                                           &redefinedProgramLink);
  const sycl::device Dev = Plt.get_devices()[0];
  current_link_options.clear();
  current_compile_options.clear();
  std::string expected_compile_options_1 =
                  "-cl-opt-disable -cl-fp32-correctly-rounded-divide-sqrt",
              expected_link_options_1 =
                  "-cl-denorms-are-zero -cl-no-signed-zeros";
  static sycl::unittest::MockDeviceImage DevImage_1 =
      generateEAMTestKernelImage<EAMTestKernel2>(expected_compile_options_1,
                                                 expected_link_options_1);

  static sycl::unittest::MockDeviceImageArray<1> DevImageArray = {&DevImage_1};
  auto KernelID_1 = sycl::get_kernel_id<EAMTestKernel2>();
  sycl::queue Queue{Dev};
  const sycl::context Ctx = Queue.get_context();
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {KernelID_1});
  auto BundleObj = sycl::compile(KernelBundle);
  sycl::link(BundleObj);
  EXPECT_EQ(expected_link_options_1, current_link_options);
  EXPECT_EQ(expected_compile_options_1, current_compile_options);
}

// According to kernel_bundle_impl.hpp:205 sycl::link now is not linking
// any two device images together
// TODO : Add check for linking 2 device images together when implemented.

TEST(Link_Compile_Options, check_sycl_build) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_before_callback("urProgramCompileExp",
                                           &redefinedProgramCompile);
  mock::getCallbacks().set_before_callback("urProgramLinkExp",
                                           &redefinedProgramLink);
  mock::getCallbacks().set_before_callback("urProgramBuildExp",
                                           &redefinedProgramBuild);
  const sycl::device Dev = Plt.get_devices()[0];
  current_link_options.clear();
  current_compile_options.clear();
  std::string expected_compile_options = "-cl-opt-disable",
              expected_link_options = "-cl-denorms-are-zero";
  static sycl::unittest::MockDeviceImage DevImage =
      generateEAMTestKernelImage<EAMTestKernel3>(expected_compile_options,
                                                 expected_link_options);
  static sycl::unittest::MockDeviceImageArray<1> DevImageArray{&DevImage};
  auto KernelID = sycl::get_kernel_id<EAMTestKernel3>();
  sycl::context Ctx{Dev};
  sycl::queue Queue{Ctx, Dev};
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {KernelID});
  sycl::build(KernelBundle);
  EXPECT_EQ(expected_compile_options + " " + expected_link_options,
            current_build_opts);
}
