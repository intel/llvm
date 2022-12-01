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

#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

class BuildOptsTestKernel;

static std::string BuildOpts;
namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
template <> struct KernelInfo<BuildOptsTestKernel> {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return "BuildOptsTestKernel"; }
  static constexpr bool isESIMD() { return true; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
  static constexpr int64_t getKernelSize() { return 1; }
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

static pi_result redefinedProgramBuild(
    pi_program prog, pi_uint32, const pi_device *, const char *options,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data) {
  if (options)
    BuildOpts = options;
  else
    BuildOpts = "";
  if (pfn_notify) {
    pfn_notify(prog, user_data);
  }
  return PI_SUCCESS;
}

static pi_result redefinedProgramCompile(pi_program, pi_uint32,
                                         const pi_device *, const char *options,
                                         pi_uint32, const pi_program *,
                                         const char **,
                                         void (*)(pi_program, void *), void *) {
  if (options)
    BuildOpts = options;
  else
    BuildOpts = "";
  return PI_SUCCESS;
}

static pi_result redefinedProgramLink(pi_context, pi_uint32, const pi_device *,
                                      const char *options, pi_uint32,
                                      const pi_program *,
                                      void (*)(pi_program, void *), void *,
                                      pi_program *) {
  if (options)
    BuildOpts = options;
  else
    BuildOpts = "";
  return PI_SUCCESS;
}

static void setupCommonMockAPIs(sycl::unittest::PiMock &Mock) {
  using namespace sycl::detail;
  Mock.redefineBefore<PiApiKind::piProgramCompile>(redefinedProgramCompile);
  Mock.redefineBefore<PiApiKind::piProgramLink>(redefinedProgramLink);
  Mock.redefineBefore<PiApiKind::piProgramBuild>(redefinedProgramBuild);
}

static sycl::unittest::PiImage generateDefaultImage() {
  using namespace sycl::unittest;

  PiPropertySet PropSet;
  addESIMDFlag(PropSet);
  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  PiArray<PiOffloadEntry> Entries = makeEmptyKernels({"BuildOptsTestKernel"});

  PiImage Img{PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
              "-compile-img",                         // Compile options
              "-link-img",                            // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}

sycl::unittest::PiImage Img = generateDefaultImage();
sycl::unittest::PiImageArray<1> ImgArray{&Img};

TEST(KernelBuildOptions, KernelBundleBasic) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  setupCommonMockAPIs(Mock);

  const sycl::device Dev = Plt.get_devices()[0];

  sycl::queue Queue{Dev};

  const sycl::context Ctx = Queue.get_context();

  auto KernelID = sycl::get_kernel_id<BuildOptsTestKernel>();
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                         {KernelID});
  auto ExecBundle = sycl::build(KernelBundle);
  EXPECT_EQ(BuildOpts,
            "-compile-img -vc-codegen -disable-finalizer-msg -link-img");

  auto ObjBundle = sycl::compile(KernelBundle, KernelBundle.get_devices());
  EXPECT_EQ(BuildOpts, "-compile-img -vc-codegen -disable-finalizer-msg");

  auto LinkBundle = sycl::link(ObjBundle, ObjBundle.get_devices());
  EXPECT_EQ(BuildOpts, "-link-img");
}
