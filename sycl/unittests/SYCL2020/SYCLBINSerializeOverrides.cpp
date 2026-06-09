//==--- SYCLBINSerializeOverrides.cpp - Override-pass tests ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Tests for the runtime-overlay (override) pass added to SYCLBIN
// serialization. Each override edits the abstract-module property registry
// after the raw forward pass; these tests pin down its observable behavior
// for end-to-end ext_oneapi_get_content paths.

#include <detail/kernel_bundle_impl.hpp>
#include <detail/syclbin.hpp>
#include <sycl/sycl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

class SYCLBINOverridesKernelA;
class SYCLBINOverridesKernelB;
MOCK_INTEGRATION_HEADER(SYCLBINOverridesKernelA)
MOCK_INTEGRATION_HEADER(SYCLBINOverridesKernelB)

namespace {

sycl::unittest::MockDeviceImage makeImage() {
  using namespace sycl::unittest;
  std::vector<unsigned char> Bin{0x01, 0x02, 0x03, 0x04};
  std::vector<MockOffloadEntry> Entries =
      makeEmptyKernels({"SYCLBINOverridesKernelA", "SYCLBINOverridesKernelB"});
  return MockDeviceImage{SYCL_DEVICE_BINARY_TYPE_NATIVE,
                         __SYCL_DEVICE_BINARY_TARGET_SPIRV64,
                         /*CompileOptions=*/"",
                         /*LinkOptions=*/"",
                         std::move(Bin),
                         std::move(Entries),
                         MockPropertySet{}};
}

sycl::unittest::MockDeviceImage Imgs[] = {makeImage()};
sycl::unittest::MockDeviceImageArray<std::size(Imgs)> ImgArray{Imgs};

// Find a property by name in the [SYCL/kernel names] property set of any
// abstract module.
bool hasKernelName(const sycl::detail::SYCLBIN &Parsed,
                   std::string_view Name) {
  for (const auto &AM : Parsed.AbstractModules) {
    if (!AM.Metadata)
      continue;
    auto It = AM.Metadata->getPropSets().find("SYCL/kernel names");
    if (It == AM.Metadata->getPropSets().end())
      continue;
    if (It->second.find(std::string{Name}) != It->second.end())
      return true;
  }
  return false;
}

} // namespace

// Override: [SYCL/kernel names] must include the requested kernel even when
// the static image carries no kernel-name property. Two kernels share a
// single image in this mock; an executable bundle scoped to one of them
// causes the image to be selected and the override emits the image's full
// kernel set.
TEST(SYCLBINSerializeOverrides, KernelNamesPresentForScopedBundle) {
  sycl::unittest::UrMock<> Mock;

  const sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};

  std::vector<sycl::kernel_id> KIDs{
      sycl::get_kernel_id<SYCLBINOverridesKernelA>()};
  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev},
                                                                    KIDs);

  std::vector<char> Bytes =
      sycl::detail::getSyclObjImpl(KB)->ext_oneapi_get_content();
  ASSERT_FALSE(Bytes.empty());

  sycl::detail::SYCLBIN Parsed{Bytes.data(), Bytes.size()};
  EXPECT_TRUE(hasKernelName(Parsed, "SYCLBINOverridesKernelA"));
}

// Override: when the bundle carries multiple kernel ids, all of them must
// appear in [SYCL/kernel names] after serialization. Guards against the
// override clearing the property set to a smaller view than the bundle's
// runtime registration.
TEST(SYCLBINSerializeOverrides, KernelNamesAllPresent) {
  sycl::unittest::UrMock<> Mock;

  const sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};

  std::vector<sycl::kernel_id> KIDs{
      sycl::get_kernel_id<SYCLBINOverridesKernelA>(),
      sycl::get_kernel_id<SYCLBINOverridesKernelB>()};
  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev},
                                                                    KIDs);

  std::vector<char> Bytes =
      sycl::detail::getSyclObjImpl(KB)->ext_oneapi_get_content();
  sycl::detail::SYCLBIN Parsed{Bytes.data(), Bytes.size()};
  EXPECT_TRUE(hasKernelName(Parsed, "SYCLBINOverridesKernelA"));
  EXPECT_TRUE(hasKernelName(Parsed, "SYCLBINOverridesKernelB"));
}
