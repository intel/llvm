//==------- SYCLBINSerializeMulti.cpp - Multi-image SYCLBIN serialize tests
//-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Companion to SYCLBINSerialize.cpp covering kernel_bundles whose underlying
// program manager registration includes multiple device images. Each image
// must be packaged into its own abstract module by ext_oneapi_get_content.

#include <detail/kernel_bundle_impl.hpp>
#include <detail/syclbin.hpp>
#include <sycl/sycl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

class SYCLBINSerializeMultiKernelA;
class SYCLBINSerializeMultiKernelB;
class SYCLBINSerializeMultiKernelC;
MOCK_INTEGRATION_HEADER(SYCLBINSerializeMultiKernelA)
MOCK_INTEGRATION_HEADER(SYCLBINSerializeMultiKernelB)
MOCK_INTEGRATION_HEADER(SYCLBINSerializeMultiKernelC)

namespace {

sycl::unittest::MockDeviceImage makeImage(const std::string &KernelName,
                                          uint8_t Format) {
  using namespace sycl::unittest;
  std::vector<unsigned char> Bin{0x01, 0x02, 0x03, 0x04};
  std::vector<MockOffloadEntry> Entries = makeEmptyKernels({KernelName});
  return MockDeviceImage{Format,
                         __SYCL_DEVICE_BINARY_TARGET_SPIRV64,
                         /*CompileOptions=*/"",
                         /*LinkOptions=*/"",
                         std::move(Bin),
                         std::move(Entries),
                         MockPropertySet{}};
}

sycl::unittest::MockDeviceImage Imgs[] = {
    makeImage("SYCLBINSerializeMultiKernelA", SYCL_DEVICE_BINARY_TYPE_SPIRV),
    makeImage("SYCLBINSerializeMultiKernelB", SYCL_DEVICE_BINARY_TYPE_SPIRV),
    makeImage("SYCLBINSerializeMultiKernelC", SYCL_DEVICE_BINARY_TYPE_SPIRV)};
sycl::unittest::MockDeviceImageArray<std::size(Imgs)> ImgArray{Imgs};

} // namespace

// A bundle covering multiple device images must produce a SYCLBIN with one
// abstract module per image. Each abstract module has exactly one IR module
// (since all images are SPIR-V here).
TEST(SYCLBINSerializeMulti, OneAbstractModulePerImage) {
  sycl::unittest::UrMock<> Mock;

  const sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};

  std::vector<sycl::kernel_id> KIDs{
      sycl::get_kernel_id<SYCLBINSerializeMultiKernelA>(),
      sycl::get_kernel_id<SYCLBINSerializeMultiKernelB>(),
      sycl::get_kernel_id<SYCLBINSerializeMultiKernelC>()};
  // Use input-state bundle so that SPIR-V images are preserved as IR modules.
  // Executable-state bundles trigger native binary extraction via UR.
  auto KB =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev}, KIDs);

  std::vector<char> Bytes =
      sycl::detail::getSyclObjImpl(KB)->ext_oneapi_get_content();
  ASSERT_FALSE(Bytes.empty());

  sycl::detail::SYCLBIN Parsed{Bytes.data(), Bytes.size()};
  EXPECT_EQ(Parsed.AbstractModules.size(), 3u);

  size_t TotalIRModules = 0;
  size_t TotalNativeImages = 0;
  for (const auto &AM : Parsed.AbstractModules) {
    EXPECT_EQ(AM.IRModules.size() + AM.NativeDeviceCodeImages.size(), 1u)
        << "Each abstract module is expected to wrap exactly one image.";
    TotalIRModules += AM.IRModules.size();
    TotalNativeImages += AM.NativeDeviceCodeImages.size();
  }
  EXPECT_EQ(TotalIRModules, 3u);
  EXPECT_EQ(TotalNativeImages, 0u);
}
