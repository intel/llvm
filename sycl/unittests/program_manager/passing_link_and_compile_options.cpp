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

class EAMTestKernel;
const char EAMTestKernelName[] = "LinkCompileTestKernel";
constexpr unsigned EAMTestKernelNumArgs = 4;

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
template <> struct KernelInfo<EAMTestKernel> {
  static constexpr unsigned getNumParams() { return EAMTestKernelNumArgs; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return EAMTestKernelName; }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

template <typename T>
static sycl::unittest::PiImage
generateEAMTestKernelImage(std::string _cmplOptions, std::string _lnkOptions) {
  using namespace sycl::unittest;

  std::vector<unsigned char> KernelEAM{0b00000101};
  PiProperty EAMKernelPOI = makeKernelParamOptInfo(
      sycl::detail::KernelInfo<T>::getName(), EAMTestKernelNumArgs, KernelEAM);
  PiArray<PiProperty> ImgKPOI{std::move(EAMKernelPOI)};

  PiPropertySet PropSet;
  PropSet.insert(__SYCL_PI_PROPERTY_SET_KERNEL_PARAM_OPT_INFO,
                 std::move(ImgKPOI));

  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  PiArray<PiOffloadEntry> Entries = makeEmptyKernels({EAMTestKernelName});

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

TEST(Link_Compile_Options, compile_link_Options_Test_empty) {
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
  std::string expected_compile_options = "";
  std::string expected_link_options = "";
  static sycl::unittest::PiImage DevImage =
      generateEAMTestKernelImage<EAMTestKernel>(expected_compile_options,
                                                expected_link_options);
  static sycl::unittest::PiImageArray<1> DevImageArray{&DevImage};
  auto KernelID = sycl::get_kernel_id<EAMTestKernel>();

  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {KernelID});

  auto BundleObj = sycl::compile(KernelBundle);
  sycl::link(BundleObj);
  EXPECT_EQ(expected_link_options, current_link_options);
  EXPECT_EQ(expected_compile_options, current_compile_options);
}

TEST(Link_Compile_Options, one_link_option_Test) {
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
  std::string expected_compile_options = "";
  std::string expected_link_options = "-cl-denorms-are-zero";
  static sycl::unittest::PiImage DevImage =
      generateEAMTestKernelImage<EAMTestKernel>(expected_compile_options,
                                                expected_link_options);
  static sycl::unittest::PiImageArray<1> DevImageArray{&DevImage};
  auto KernelID = sycl::get_kernel_id<EAMTestKernel>();

  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {KernelID});

  auto BundleObj = sycl::compile(KernelBundle);
  sycl::link(BundleObj);
  EXPECT_EQ(expected_link_options, current_link_options);
  EXPECT_EQ(expected_compile_options, current_compile_options);
}

TEST(Link_Compile_Options, one_compile_option_Test) {
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
  std::string expected_compile_options = "-cl-single-precision-constant";
  std::string expected_link_options = "";
  static sycl::unittest::PiImage DevImage =
      generateEAMTestKernelImage<EAMTestKernel>(expected_compile_options,
                                                expected_link_options);
  static sycl::unittest::PiImageArray<1> DevImageArray{&DevImage};
  auto KernelID = sycl::get_kernel_id<EAMTestKernel>();

  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {KernelID});

  auto BundleObj = sycl::compile(KernelBundle);
  sycl::link(BundleObj);
  EXPECT_EQ(expected_link_options, current_link_options);
  EXPECT_EQ(expected_compile_options, current_compile_options);
}

TEST(Link_Compile_Options, one_link_and_compile_option_Test) {
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
  std::string expected_compile_options = "-cl-single-precision-constant";
  std::string expected_link_options = "-cl-finite-math-only";
  static sycl::unittest::PiImage DevImage =
      generateEAMTestKernelImage<EAMTestKernel>(expected_compile_options,
                                                expected_link_options);
  static sycl::unittest::PiImageArray<1> DevImageArray{&DevImage};
  auto KernelID = sycl::get_kernel_id<EAMTestKernel>();

  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {KernelID});

  auto BundleObj = sycl::compile(KernelBundle);
  sycl::link(BundleObj);
  EXPECT_EQ(expected_link_options, current_link_options);
  EXPECT_EQ(expected_compile_options, current_compile_options);
}