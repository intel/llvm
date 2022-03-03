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

std::string current_link_options, current_compile_options, current_build_opts;

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
  if (!current_link_options.empty())
    current_link_options += " ";
  current_link_options += std::string(_linkOpts);
  return PI_SUCCESS;
}

inline pi_result redefinedProgramCompile(pi_program, pi_uint32,
                                         const pi_device *,
                                         const char *_compileOpts, pi_uint32,
                                         const pi_program *, const char **,
                                         void (*)(pi_program, void *), void *) {
  assert(_compileOpts != nullptr);
  if (!current_compile_options.empty())
    current_compile_options += " ";
  current_compile_options += std::string(_compileOpts);
  return PI_SUCCESS;
}

inline pi_result redefinedProgramBuild(
    pi_program prog, pi_uint32, const pi_device *, const char *options,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data) {
  assert(options != nullptr);
  current_build_opts = std::string(options);
  return PI_SUCCESS;
}

TEST(Link_Compile_Options, compile_link_Options_Test_empty_options) {
  sycl::platform Plt{sycl::default_selector()};
  if (Plt.is_host()) {
    std::cerr << "Test is not supported on host, skipping\n";
    GTEST_SKIP(); // test is not supported on host.
  }

  if (Plt.get_backend() == sycl::backend::ext_oneapi_cuda) {
    std::cerr << "Test is not supported on CUDA platform, skipping\n";
    GTEST_SKIP();
  }

  if (Plt.get_backend() == sycl::backend::ext_oneapi_hip) {
    std::cerr << "Test is not supported on HIP platform, skipping\n";
    GTEST_SKIP();
  }
  sycl::unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  Mock.redefine<sycl::detail::PiApiKind::piProgramCompile>(
      redefinedProgramCompile);
  Mock.redefine<sycl::detail::PiApiKind::piProgramLink>(redefinedProgramLink);
  const sycl::device Dev = Plt.get_devices()[0];
  current_link_options.clear();
  current_compile_options.clear();
  std::string expected_options = "";
  static sycl::unittest::PiImage DevImage =
      generateEAMTestKernel1Image<EAMTestKernel1>(expected_options,
                                                  expected_options);
  static sycl::unittest::PiImageArray<1> DevImageArray_{&DevImage};
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
  sycl::platform Plt{sycl::default_selector()};
  if (Plt.is_host()) {
    std::cerr << "Test is not supported on host, skipping\n";
    GTEST_SKIP(); // test is not supported on host.
  }

  if (Plt.get_backend() == sycl::backend::ext_oneapi_cuda) {
    std::cerr << "Test is not supported on CUDA platform, skipping\n";
    GTEST_SKIP();
  }

  if (Plt.get_backend() == sycl::backend::ext_oneapi_hip) {
    std::cerr << "Test is not supported on HIP platform, skipping\n";
    GTEST_SKIP();
  }
  sycl::unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  Mock.redefine<sycl::detail::PiApiKind::piProgramCompile>(
      redefinedProgramCompile);
  Mock.redefine<sycl::detail::PiApiKind::piProgramLink>(redefinedProgramLink);
  const sycl::device Dev = Plt.get_devices()[0];
  current_link_options.clear();
  current_compile_options.clear();
  std::string expected_compile_options_1 =
                  "-cl-opt-disable -cl-fp32-correctly-rounded-divide-sqrt",
              expected_link_options_1 =
                  "-cl-denorms-are-zero -cl-no-signed-zeros";
  static sycl::unittest::PiImage DevImage_1 =
      generateEAMTestKernel1Image<EAMTestKernel1>(expected_compile_options_1,
                                                  expected_link_options_1);

  static sycl::unittest::PiImageArray<1> DevImageArray = {&DevImage_1};
  auto KernelID_1 = sycl::get_kernel_id<EAMTestKernel1>();
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
  sycl::platform Plt{sycl::default_selector()};
  if (Plt.is_host()) {
    std::cerr << "Test is not supported on host, skipping\n";
    GTEST_SKIP(); // test is not supported on host.
  }

  if (Plt.get_backend() == sycl::backend::ext_oneapi_cuda) {
    std::cerr << "Test is not supported on CUDA platform, skipping\n";
    GTEST_SKIP();
  }

  if (Plt.get_backend() == sycl::backend::ext_oneapi_hip) {
    std::cerr << "Test is not supported on HIP platform, skipping\n";
    GTEST_SKIP();
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
  std::string expected_compile_options = "-cl-opt-disable",
              expected_link_options = "-cl-denorms-are-zero";
  static sycl::unittest::PiImage DevImage =
      generateEAMTestKernel1Image<EAMTestKernel1>(expected_compile_options,
                                                  expected_link_options);
  static sycl::unittest::PiImageArray<1> DevImageArray{&DevImage};
  auto KernelID = sycl::get_kernel_id<EAMTestKernel1>();
  sycl::queue Queue{Dev};
  const sycl::context Ctx = Queue.get_context();
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {KernelID});
  sycl::build(KernelBundle);
  EXPECT_EQ(expected_compile_options + " " + expected_link_options,
            current_build_opts);
}