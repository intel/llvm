//==---- passing_link_and_compile_options.cpp --- Test for passing link and
// compile options for online linker and compiler ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

std::string current_link_options, current_compile_options;

class EAMTestKernel1;
const char EAMTestKernelName1[] = "LinkCompileTestKernel1";
constexpr unsigned EAMTestKernelNumArgs1 = 4;

class EAMTestKernel2;
const char EAMTestKernelName2[] = "LinkCompileTestKernel2";
constexpr unsigned EAMTestKernelNumArgs2 = 4;

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
template <> struct KernelInfo<EAMTestKernel1> {
  static constexpr unsigned getNumParams() { return EAMTestKernelNumArgs1; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return EAMTestKernelName1; }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
};

template <> struct KernelInfo<EAMTestKernel2> {
  static constexpr unsigned getNumParams() { return EAMTestKernelNumArgs2; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return EAMTestKernelName2; }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

template <typename T>
static sycl::unittest::PiImage
generateEAMTestKernel1Image(std::string _cmplOptions, std::string _lnkOptions) {
  using namespace sycl::unittest;

  std::vector<unsigned char> KernelEAM1{0b00000101};
  PiProperty EAMKernelPOI =
      makeKernelParamOptInfo(sycl::detail::KernelInfo<T>::getName(),
                             EAMTestKernelNumArgs1, KernelEAM1);
  PiArray<PiProperty> ImgKPOI{std::move(EAMKernelPOI)};

  PiPropertySet PropSet;
  PropSet.insert(__SYCL_PI_PROPERTY_SET_KERNEL_PARAM_OPT_INFO,
                 std::move(ImgKPOI));

  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  PiArray<PiOffloadEntry> Entries = makeEmptyKernels({EAMTestKernelName1});

  PiImage Img{PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
              _cmplOptions,                           // Compile options
              _lnkOptions,                            // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};
  return Img;
}

template <typename T>
static sycl::unittest::PiImage
generateEAMTestKernel2Image(std::string _cmplOptions, std::string _lnkOptions) {
  using namespace sycl::unittest;

  std::vector<unsigned char> KernelEAM2{0b00000101};
  PiProperty EAMKernelPOI =
      makeKernelParamOptInfo(sycl::detail::KernelInfo<T>::getName(),
                             EAMTestKernelNumArgs2, KernelEAM2);
  PiArray<PiProperty> ImgKPOI{std::move(EAMKernelPOI)};

  PiPropertySet PropSet;
  PropSet.insert(__SYCL_PI_PROPERTY_SET_KERNEL_PARAM_OPT_INFO,
                 std::move(ImgKPOI));

  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  PiArray<PiOffloadEntry> Entries = makeEmptyKernels({EAMTestKernelName2});

  PiImage Img{PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
              _cmplOptions,                           // Compile options
              _lnkOptions,                            // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};
  return Img;
}

inline pi_result redefinedProgramLink(pi_context, pi_uint32, const pi_device *,
                                      const char *_linkOpts, pi_uint32,
                                      const pi_program *,
                                      void (*)(pi_program, void *), void *,
                                      pi_program *) {
  assert(_linkOpts != nullptr);
  current_link_options = std::string(_linkOpts);
  return PI_SUCCESS;
}

inline pi_result redefinedProgramCompile(pi_program, pi_uint32,
                                         const pi_device *,
                                         const char *_compileOpts, pi_uint32,
                                         const pi_program *, const char **,
                                         void (*)(pi_program, void *), void *) {
  assert(_compileOpts != nullptr);
  current_compile_options = std::string(_compileOpts);
  return PI_SUCCESS;
}

inline pi_result redefinedProgramBuild(
    pi_program prog, pi_uint32, const pi_device *, const char *options,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data) {
  return PI_SUCCESS;
}

TEST(Link_Compile_Options, compile_link_Options_Test_empty_options) {
  sycl::platform Plt{sycl::default_selector()};
  if (Plt.is_host()) {
    std::cerr << "Test is not supported on host, skipping\n";
    return; // test is not supported on host.
  }

  if (Plt.get_backend() == sycl::backend::ext_oneapi_cuda) {
    std::cerr << "Test is not supported on CUDA platform, skipping\n";
    return;
  }

  if (Plt.get_backend() == sycl::backend::ext_oneapi_hip) {
    std::cerr << "Test is not supported on HIP platform, skipping\n";
    return;
  }
  sycl::unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  Mock.redefine<sycl::detail::PiApiKind::piProgramCompile>(
      redefinedProgramCompile);
  Mock.redefine<sycl::detail::PiApiKind::piProgramLink>(redefinedProgramLink);

  const sycl::device Dev = Plt.get_devices()[0];
  // Check devices' compile & link options separately
  current_link_options.clear();
  current_compile_options.clear();
  std::string expected_compile_options1 = "", expected_compile_options2 = "";
  std::string expected_link_options1 = "", expected_link_options2 = "";
  static sycl::unittest::PiImage DevImage1 =
      generateEAMTestKernel1Image<EAMTestKernel1>(expected_compile_options1,
                                                  expected_link_options1);
  static sycl::unittest::PiImage DevImage2 =
      generateEAMTestKernel1Image<EAMTestKernel2>(expected_compile_options2,
                                                  expected_link_options2);
  static sycl::unittest::PiImageArray<1> DevImageArray_1{&DevImage1};
  static sycl::unittest::PiImageArray<1> DevImageArray_2{&DevImage2};
  auto KernelID_1 = sycl::get_kernel_id<EAMTestKernel1>();
  auto KernelID_2 = sycl::get_kernel_id<EAMTestKernel2>();
  sycl::queue Queue{Dev};
  const sycl::context Ctx = Queue.get_context();
  sycl::kernel_bundle KernelBundle1 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {KernelID_1});
  auto BundleObj1 = sycl::compile(KernelBundle1);
  sycl::link(BundleObj1);
  EXPECT_EQ(expected_link_options1, current_link_options);
  EXPECT_EQ(expected_compile_options1, current_compile_options);
  current_link_options.clear();
  current_compile_options.clear();
  sycl::kernel_bundle KernelBundle2 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {KernelID_2});
  auto BundleObj2 = sycl::compile(KernelBundle2);
  sycl::link(BundleObj2);
  EXPECT_EQ(expected_link_options2, current_link_options);
  EXPECT_EQ(expected_compile_options2, current_compile_options);
  // Check devices' compile & link options together
  current_link_options.clear();
  current_compile_options.clear();
  static sycl::unittest::PiImageArray<2> DevImageArray[] = {&DevImage1,
                                                            &DevImage2};
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(
          Ctx, {Dev}, {KernelID_1, KernelID_2});
  auto BundleObj = sycl::compile(KernelBundle);
  sycl::link(BundleObj);
  EXPECT_EQ("", current_link_options);
  EXPECT_EQ("", current_compile_options);
}

TEST(Link_Compile_Options, compile_link_Options_Test_only_compile_options) {
  sycl::platform Plt{sycl::default_selector()};
  if (Plt.is_host()) {
    std::cerr << "Test is not supported on host, skipping\n";
    return; // test is not supported on host.
  }

  if (Plt.get_backend() == sycl::backend::ext_oneapi_cuda) {
    std::cerr << "Test is not supported on CUDA platform, skipping\n";
    return;
  }

  if (Plt.get_backend() == sycl::backend::ext_oneapi_hip) {
    std::cerr << "Test is not supported on HIP platform, skipping\n";
    return;
  }
  sycl::unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  Mock.redefine<sycl::detail::PiApiKind::piProgramCompile>(
      redefinedProgramCompile);
  Mock.redefine<sycl::detail::PiApiKind::piProgramLink>(redefinedProgramLink);

  const sycl::device Dev = Plt.get_devices()[0];
  // Check devices' compile & link options separately
  current_link_options.clear();
  current_compile_options.clear();
  std::string expected_compile_options1 = "-cl-single-precision-constant",
              expected_compile_options1_1 = "",
              expected_compile_options2 =
                  "-cl-fp32-correctly-rounded-divide-sqrt",
              expected_compile_options2_1 = "";
  std::string expected_link_options = "";
  static sycl::unittest::PiImage DevImage1 =
      generateEAMTestKernel1Image<EAMTestKernel1>(expected_compile_options1,
                                                  expected_link_options);
  static sycl::unittest::PiImage DevImage2 =
      generateEAMTestKernel1Image<EAMTestKernel2>(expected_compile_options2,
                                                  expected_link_options);
  static sycl::unittest::PiImageArray<1> DevImageArray_1{&DevImage1};
  static sycl::unittest::PiImageArray<1> DevImageArray_2{&DevImage2};
  auto KernelID_1 = sycl::get_kernel_id<EAMTestKernel1>();
  auto KernelID_2 = sycl::get_kernel_id<EAMTestKernel2>();
  sycl::queue Queue{Dev};
  const sycl::context Ctx = Queue.get_context();
  sycl::kernel_bundle KernelBundle1 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {KernelID_1});
  auto BundleObj1 = sycl::compile(KernelBundle1);
  sycl::link(BundleObj1);
  EXPECT_EQ(expected_link_options, current_link_options);
  EXPECT_EQ(expected_compile_options1, current_compile_options);
  current_link_options.clear();
  current_compile_options.clear();
  sycl::kernel_bundle KernelBundle2 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {KernelID_2});
  auto BundleObj2 = sycl::compile(KernelBundle2);
  sycl::link(BundleObj2);
  EXPECT_EQ(expected_compile_options2, current_compile_options);
  EXPECT_EQ(expected_link_options, current_link_options);
  // Check devices' compile & link options together
  current_link_options.clear();
  current_compile_options.clear();
  static sycl::unittest::PiImageArray<2> DevImageArray[] = {&DevImage1,
                                                            &DevImage2};
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(
          Ctx, {Dev}, {KernelID_1, KernelID_2});
  auto BundleObj = sycl::compile(KernelBundle);
  sycl::link(BundleObj);
  EXPECT_EQ("", current_link_options);
  EXPECT_EQ(expected_compile_options1 + " " + expected_compile_options2,
            current_compile_options);
  // empty str + not empty str
  DevImage1 = generateEAMTestKernel1Image<EAMTestKernel1>(
      expected_compile_options1_1, expected_link_options);
  current_link_options.clear();
  current_compile_options.clear();
  static sycl::unittest::PiImageArray<2> DevImageArray1[] = {&DevImage1,
                                                             &DevImage2};

  BundleObj = sycl::compile(KernelBundle);
  sycl::link(BundleObj);
  EXPECT_EQ("", current_link_options);
  EXPECT_EQ(expected_compile_options2, current_compile_options);
  // not empty str + empty
  DevImage1 = generateEAMTestKernel1Image<EAMTestKernel1>(
      expected_compile_options1, expected_link_options);
  DevImage2 = generateEAMTestKernel1Image<EAMTestKernel2>(
      expected_compile_options2_1, expected_link_options);
  current_link_options.clear();
  current_compile_options.clear();
  static sycl::unittest::PiImageArray<2> DevImageArray2[] = {&DevImage1,
                                                             &DevImage2};

  BundleObj = sycl::compile(KernelBundle);
  sycl::link(BundleObj);
  EXPECT_EQ("", current_link_options);
  EXPECT_EQ(expected_compile_options1, current_compile_options);
}

TEST(Link_Compile_Options, compile_link_Options_Test_only_link_options) {
  sycl::platform Plt{sycl::default_selector()};
  if (Plt.is_host()) {
    std::cerr << "Test is not supported on host, skipping\n";
    return; // test is not supported on host.
  }

  if (Plt.get_backend() == sycl::backend::ext_oneapi_cuda) {
    std::cerr << "Test is not supported on CUDA platform, skipping\n";
    return;
  }

  if (Plt.get_backend() == sycl::backend::ext_oneapi_hip) {
    std::cerr << "Test is not supported on HIP platform, skipping\n";
    return;
  }
  sycl::unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  Mock.redefine<sycl::detail::PiApiKind::piProgramCompile>(
      redefinedProgramCompile);
  Mock.redefine<sycl::detail::PiApiKind::piProgramLink>(redefinedProgramLink);

  const sycl::device Dev = Plt.get_devices()[0];
  // Check devices' compile & link options separately
  current_link_options.clear();
  current_compile_options.clear();
  std::string expected_compile_options = "";
  std::string expected_link_options1 = "-cl-finite-math-only",
              expected_link_options1_1 = "",
              expected_link_options2 = "-cl-no-signed-zeros",
              expected_link_options2_1 = "";
  static sycl::unittest::PiImage DevImage1 =
      generateEAMTestKernel1Image<EAMTestKernel1>(expected_compile_options,
                                                  expected_link_options1);
  static sycl::unittest::PiImage DevImage2 =
      generateEAMTestKernel1Image<EAMTestKernel2>(expected_compile_options,
                                                  expected_link_options2);
  static sycl::unittest::PiImageArray<1> DevImageArray_1{&DevImage1};
  static sycl::unittest::PiImageArray<1> DevImageArray_2{&DevImage2};
  auto KernelID_1 = sycl::get_kernel_id<EAMTestKernel1>();
  auto KernelID_2 = sycl::get_kernel_id<EAMTestKernel2>();
  sycl::queue Queue{Dev};
  const sycl::context Ctx = Queue.get_context();
  sycl::kernel_bundle KernelBundle1 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {KernelID_1});
  auto BundleObj1 = sycl::compile(KernelBundle1);
  sycl::link(BundleObj1);
  EXPECT_EQ(expected_link_options1, current_link_options);
  EXPECT_EQ(expected_compile_options, current_compile_options);
  current_link_options.clear();
  current_compile_options.clear();
  sycl::kernel_bundle KernelBundle2 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {KernelID_2});
  auto BundleObj2 = sycl::compile(KernelBundle2);
  sycl::link(BundleObj2);
  EXPECT_EQ(expected_compile_options, current_compile_options);
  EXPECT_EQ(expected_link_options2, current_link_options);
  // Check devices' compile & link options together
  current_link_options.clear();
  current_compile_options.clear();
  static sycl::unittest::PiImageArray<2> DevImageArray[] = {&DevImage1,
                                                            &DevImage2};
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(
          Ctx, {Dev}, {KernelID_1, KernelID_2});
  auto BundleObj = sycl::compile(KernelBundle);
  sycl::link(BundleObj);
  EXPECT_EQ(expected_link_options1 + " " + expected_link_options2,
            current_link_options);
  EXPECT_EQ("", current_compile_options);
  // empty str + not empty str
  DevImage1 = generateEAMTestKernel1Image<EAMTestKernel1>(
      expected_compile_options, expected_link_options1_1);
  current_link_options.clear();
  current_compile_options.clear();
  static sycl::unittest::PiImageArray<2> DevImageArray1[] = {&DevImage1,
                                                             &DevImage2};

  BundleObj = sycl::compile(KernelBundle);
  sycl::link(BundleObj);
  EXPECT_EQ(expected_link_options2, current_link_options);
  EXPECT_EQ("", current_compile_options);
  // not empty str + empty
  DevImage1 = generateEAMTestKernel1Image<EAMTestKernel1>(
      expected_compile_options, expected_link_options1);
  DevImage2 = generateEAMTestKernel1Image<EAMTestKernel2>(
      expected_compile_options, expected_link_options2_1);
  current_link_options.clear();
  current_compile_options.clear();
  static sycl::unittest::PiImageArray<2> DevImageArray2[] = {&DevImage1,
                                                             &DevImage2};

  BundleObj = sycl::compile(KernelBundle);
  sycl::link(BundleObj);
  EXPECT_EQ(expected_link_options1, current_link_options);
  EXPECT_EQ("", current_compile_options);
}

TEST(Link_Compile_Options, link_compile_options_all_options) {
  sycl::platform Plt{sycl::default_selector()};
  if (Plt.is_host()) {
    std::cerr << "Test is not supported on host, skipping\n";
    return; // test is not supported on host.
  }

  if (Plt.get_backend() == sycl::backend::ext_oneapi_cuda) {
    std::cerr << "Test is not supported on CUDA platform, skipping\n";
    return;
  }

  if (Plt.get_backend() == sycl::backend::ext_oneapi_hip) {
    std::cerr << "Test is not supported on HIP platform, skipping\n";
    return;
  }
  sycl::unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  Mock.redefine<sycl::detail::PiApiKind::piProgramCompile>(
      redefinedProgramCompile);
  Mock.redefine<sycl::detail::PiApiKind::piProgramLink>(redefinedProgramLink);

  const sycl::device Dev = Plt.get_devices()[0];

  current_link_options.clear();
  current_compile_options.clear();
  std::string expected_compile_options1 = "-cl-single-precision-constant",
              expected_compile_options2 =
                  "-cl-fp32-correctly-rounded-divide-sqrt";
  std::string expected_link_options1 = "-cl-finite-math-only",
              expected_link_options2 = "-cl-no-signed-zeros";
  static sycl::unittest::PiImage DevImage1 =
      generateEAMTestKernel1Image<EAMTestKernel1>(expected_compile_options1,
                                                  expected_link_options1);
  static sycl::unittest::PiImage DevImage2 =
      generateEAMTestKernel2Image<EAMTestKernel2>(expected_compile_options2,
                                                  expected_link_options2);
  static sycl::unittest::PiImageArray<2> DevImageArray[] = {&DevImage1,
                                                            &DevImage2};
  auto KernelID_1 = sycl::get_kernel_id<EAMTestKernel1>();
  auto KernelID_2 = sycl::get_kernel_id<EAMTestKernel2>();
  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();
  // first device image check
  sycl::kernel_bundle KernelBundle1 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {KernelID_1});
  auto BundleObj1 = sycl::compile(KernelBundle1);
  sycl::link(BundleObj1);
  EXPECT_EQ(expected_link_options1, current_link_options);
  EXPECT_EQ(expected_compile_options1, current_compile_options);
  // second device image check
  current_link_options.clear();
  current_compile_options.clear();
  sycl::kernel_bundle KernelBundle2 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {KernelID_2});
  auto BundleObj2 = sycl::compile(KernelBundle2);
  sycl::link(BundleObj2);
  EXPECT_EQ(expected_link_options2, current_link_options);
  EXPECT_EQ(expected_compile_options2, current_compile_options);
  // 2 device images check
  current_link_options.clear();
  current_compile_options.clear();
  auto KernelBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
      Ctx, {Dev}, {KernelID_1, KernelID_2});

  auto BundleObj = sycl::compile(KernelBundle);
  sycl::link(BundleObj);
  EXPECT_EQ(expected_link_options1 + " " + expected_link_options2,
            current_link_options);
  EXPECT_EQ(expected_compile_options1 + " " + expected_compile_options2,
            current_compile_options);
}

TEST(Link_Compile_Options, sycl_build_test) {
  sycl::platform Plt{sycl::default_selector()};
  if (Plt.is_host()) {
    std::cerr << "Test is not supported on host, skipping\n";
    return; // test is not supported on host.
  }

  if (Plt.get_backend() == sycl::backend::ext_oneapi_cuda) {
    std::cerr << "Test is not supported on CUDA platform, skipping\n";
    return;
  }

  if (Plt.get_backend() == sycl::backend::ext_oneapi_hip) {
    std::cerr << "Test is not supported on HIP platform, skipping\n";
    return;
  }
  sycl::unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  Mock.redefine<sycl::detail::PiApiKind::piProgramCompile>(
      redefinedProgramCompile);
  Mock.redefine<sycl::detail::PiApiKind::piProgramLink>(redefinedProgramLink);
  Mock.redefine<sycl::detail::PiApiKind::piProgramBuild>(redefinedProgramBuild);
  const sycl::device Dev = Plt.get_devices()[0];
  current_link_options.clear();
  current_compile_options.clear();
  std::string expected_compile_options = "-cl-single-precision-constant";
  std::string expected_link_options = "-cl-finite-math-only";
  static sycl::unittest::PiImage DevImage =
      generateEAMTestKernel1Image<EAMTestKernel1>(expected_compile_options,
                                                  expected_link_options);
  auto KernelID_1 = sycl::get_kernel_id<EAMTestKernel1>();
  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();
  sycl::kernel_bundle KernelBundle1 =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {KernelID_1});
  auto Dummy = sycl::build(KernelBundle1);
  EXPECT_EQ(expected_link_options, current_link_options);
  EXPECT_EQ(expected_compile_options, current_compile_options);
}
