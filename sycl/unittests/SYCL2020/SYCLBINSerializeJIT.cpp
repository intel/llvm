//==------- SYCLBINSerializeJIT.cpp - SYCLBIN JIT-image serialize tests ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Companion to SYCLBINSerialize.cpp covering the JIT (SPIR-V) image branch of
// SYCLBIN::serializeImages. The two test files use disjoint kernel names so
// each registers its own MockDeviceImageArray without colliding in the global
// program manager.

#include <detail/kernel_bundle_impl.hpp>
#include <detail/syclbin.hpp>
#include <sycl/sycl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

class SYCLBINSerializeJITKernel;
MOCK_INTEGRATION_HEADER(SYCLBINSerializeJITKernel)

namespace {

// JIT-style mock device image: SPIR-V format, generic spir64 target, no
// compile_target. ext_oneapi_get_content must classify this as an IR module
// (not a native device code image) and emit "SYCLBIN/ir module metadata"
// rather than "SYCLBIN/native device code image metadata".
sycl::unittest::MockDeviceImage makeSPIRVImage(const std::string &KernelName) {
  using namespace sycl::unittest;

  std::vector<unsigned char> Bin{0xCA, 0xFE, 0xBA, 0xBE};
  std::vector<MockOffloadEntry> Entries = makeEmptyKernels({KernelName});

  return MockDeviceImage{SYCL_DEVICE_BINARY_TYPE_SPIRV,
                         __SYCL_DEVICE_BINARY_TARGET_SPIRV64,
                         /*CompileOptions=*/"",
                         /*LinkOptions=*/"",
                         std::move(Bin),
                         std::move(Entries),
                         MockPropertySet{}};
}

sycl::unittest::MockDeviceImage Imgs[] = {
    makeSPIRVImage("SYCLBINSerializeJITKernel")};
sycl::unittest::MockDeviceImageArray<std::size(Imgs)> ImgArray{Imgs};

} // namespace

// Narrow the kernel_bundle to just our SPIR-V mock image to avoid contamination
// from images registered by sibling test files (e.g. SYCLBINSerialize.cpp).
sycl::kernel_bundle<sycl::bundle_state::executable>
getKBForKernel(const sycl::context &Ctx, const sycl::device &Dev) {
  std::vector<sycl::kernel_id> KIDs{
      sycl::get_kernel_id<SYCLBINSerializeJITKernel>()};
  return sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev},
                                                                 KIDs);
}

// Tests for SYCL runtime library, test plan section 10.3 (JIT branch): a SPIR-V
// device image must be packaged as an SYCLBIN IR module with the correct
// "SYCLBIN/ir module metadata" property set carrying type=0 (SPIR-V) and a
// non-empty target string.
TEST(SYCLBINSerializeJIT, SPIRVImageBecomesIRModule) {
  sycl::unittest::UrMock<> Mock;

  const sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};

  auto KB = getKBForKernel(Ctx, Dev);

  std::vector<char> Bytes =
      sycl::detail::getSyclObjImpl(KB)->ext_oneapi_get_content();
  ASSERT_FALSE(Bytes.empty());

  sycl::detail::SYCLBIN Parsed{Bytes.data(), Bytes.size()};
  EXPECT_EQ(Parsed.Version, 1u);

  // SPIR-V images must land in the IR module branch, not the native one.
  size_t TotalIRModules = 0;
  size_t TotalNativeImages = 0;
  for (const auto &AM : Parsed.AbstractModules) {
    TotalIRModules += AM.IRModules.size();
    TotalNativeImages += AM.NativeDeviceCodeImages.size();
  }
  EXPECT_GE(TotalIRModules, 1u);
  EXPECT_EQ(TotalNativeImages, 0u);

  // Validate the IR module metadata: type=0 (SPIR-V), target field set.
  bool FoundTypeSPIRV = false;
  bool FoundTarget = false;
  for (const auto &AM : Parsed.AbstractModules) {
    for (const auto &IRM : AM.IRModules) {
      ASSERT_NE(IRM.Metadata, nullptr);
      auto It = IRM.Metadata->getPropSets().find(
          sycl::detail::PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA);
      ASSERT_NE(It, IRM.Metadata->getPropSets().end());

      auto TyIt = It->second.find("type");
      ASSERT_NE(TyIt, It->second.end());
      ASSERT_EQ(TyIt->second.getType(),
                sycl::detail::PropertyValue::UINT32);
      if (TyIt->second.asUint32() == 0u)
        FoundTypeSPIRV = true;

      auto TgtIt = It->second.find("target");
      ASSERT_NE(TgtIt, It->second.end());
      ASSERT_EQ(TgtIt->second.getType(),
                sycl::detail::PropertyValue::BYTE_ARRAY);
      if (TgtIt->second.getByteArraySize() > 0)
        FoundTarget = true;
    }
  }
  EXPECT_TRUE(FoundTypeSPIRV);
  EXPECT_TRUE(FoundTarget);
}

// Tests for SYCL runtime library, test plan section 10.3 (JIT branch),
// continued: the IR payload bytes must round-trip through the binary byte
// table unchanged.
TEST(SYCLBINSerializeJIT, IRModuleBytesRoundTrip) {
  sycl::unittest::UrMock<> Mock;

  const sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};

  auto KB = getKBForKernel(Ctx, Dev);

  std::vector<char> Bytes =
      sycl::detail::getSyclObjImpl(KB)->ext_oneapi_get_content();
  sycl::detail::SYCLBIN Parsed{Bytes.data(), Bytes.size()};

  bool FoundPayload = false;
  for (const auto &AM : Parsed.AbstractModules) {
    for (const auto &IRM : AM.IRModules) {
      if (IRM.RawIRBytes.size() != 4)
        continue;
      const auto *P =
          reinterpret_cast<const unsigned char *>(IRM.RawIRBytes.data());
      if (P[0] == 0xCA && P[1] == 0xFE && P[2] == 0xBA && P[3] == 0xBE)
        FoundPayload = true;
    }
  }
  EXPECT_TRUE(FoundPayload)
      << "IR module payload bytes did not round-trip correctly.";
}
