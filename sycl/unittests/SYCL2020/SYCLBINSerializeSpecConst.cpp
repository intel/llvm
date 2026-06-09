//==----- SYCLBINSerializeSpecConst.cpp - Spec const override unit test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// End-to-end unit test for the [SYCL/specialization constants] /
// [SYCL/specialization constants default values] override pass added to the
// SYCLBIN serializer. Drives a real kernel_bundle whose underlying
// device_image_impl carries spec-const descriptor + default-value
// properties, then asserts the override emits both property sets in the
// SYCLBIN with byte-equivalent content.

#include <detail/kernel_bundle_impl.hpp>
#include <detail/syclbin.hpp>
#include <sycl/sycl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

class SYCLBINSerializeSpecConstKernel;
MOCK_INTEGRATION_HEADER(SYCLBINSerializeSpecConstKernel)

namespace {

// Synthetic spec-const descriptor: one scalar leaf, ID=42, offset=0, size=4.
// Defaults blob holds the 4-byte int value 0x11223344.
struct ScalarSpecConst {
  static constexpr uint32_t ID = 42;
  static constexpr uint32_t Offset = 0;
  static constexpr uint32_t Size = 4;
  static constexpr uint32_t DefaultValue = 0x11223344;
};

sycl::unittest::MockDeviceImage makeImageWithSpecConst() {
  using namespace sycl::unittest;

  // Build descriptor property using the mock helper.
  std::vector<char> ValData;
  MockProperty SCDescriptor = makeSpecConstant<uint32_t>(
      ValData, "my_spec_const", {ScalarSpecConst::ID},
      {ScalarSpecConst::Offset},
      std::tuple<uint32_t>{ScalarSpecConst::DefaultValue});

  MockPropertySet PropSet;
  addSpecConstants({std::move(SCDescriptor)}, std::move(ValData), PropSet);

  std::vector<unsigned char> Bin{0x01, 0x02, 0x03, 0x04};
  std::vector<MockOffloadEntry> Entries =
      makeEmptyKernels({"SYCLBINSerializeSpecConstKernel"});

  return MockDeviceImage{SYCL_DEVICE_BINARY_TYPE_NATIVE,
                         __SYCL_DEVICE_BINARY_TARGET_SPIRV64,
                         /*CompileOptions=*/"",
                         /*LinkOptions=*/"",
                         std::move(Bin),
                         std::move(Entries),
                         std::move(PropSet)};
}

sycl::unittest::MockDeviceImage Imgs[] = {makeImageWithSpecConst()};
sycl::unittest::MockDeviceImageArray<std::size(Imgs)> ImgArray{Imgs};

} // namespace

// Override: [SYCL/specialization constants] descriptor and
// [SYCL/specialization constants default values] blob must round-trip
// byte-equivalent through the override pass when the source image already
// carries them in its static property sets.
TEST(SYCLBINSerializeSpecConst, DescriptorAndDefaultValueRoundTrip) {
  using namespace sycl::detail;
  sycl::unittest::UrMock<> Mock;

  const sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};

  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev});

  std::vector<char> Bytes =
      sycl::detail::getSyclObjImpl(KB)->ext_oneapi_get_content();
  ASSERT_FALSE(Bytes.empty());

  SYCLBIN Parsed{Bytes.data(), Bytes.size()};

  // Locate the abstract module that carries our spec const.
  bool Found = false;
  for (const auto &AM : Parsed.AbstractModules) {
    ASSERT_NE(AM.Metadata, nullptr);
    auto SCIt =
        AM.Metadata->getPropSets().find("SYCL/specialization constants");
    if (SCIt == AM.Metadata->getPropSets().end())
      continue;
    auto DescIt = SCIt->second.find("my_spec_const");
    if (DescIt == SCIt->second.end())
      continue;
    Found = true;

    // Descriptor: 3 little-endian uint32_t per scalar leaf -> ID, Offset,
    // Size.
    ASSERT_EQ(DescIt->second.getType(), PropertyValue::BYTE_ARRAY);
    const PropertyValue::SizeTy DescBytes = DescIt->second.getByteArraySize();
    ASSERT_EQ(DescBytes, 3u * sizeof(uint32_t));
    uint32_t Triple[3];
    std::memcpy(Triple, DescIt->second.asByteArray(), sizeof(Triple));
    EXPECT_EQ(Triple[0], ScalarSpecConst::ID);
    EXPECT_EQ(Triple[1], ScalarSpecConst::Offset);
    EXPECT_EQ(Triple[2], ScalarSpecConst::Size);

    // Default values blob: must contain the original 4-byte payload for our
    // spec const, byte-equal.
    auto DVIt = AM.Metadata->getPropSets().find(
        "SYCL/specialization constants default values");
    ASSERT_NE(DVIt, AM.Metadata->getPropSets().end());
    auto DVPropIt = DVIt->second.find("my_spec_const");
    ASSERT_NE(DVPropIt, DVIt->second.end());
    ASSERT_EQ(DVPropIt->second.getType(), PropertyValue::BYTE_ARRAY);
    EXPECT_EQ(DVPropIt->second.getByteArraySize(),
              static_cast<PropertyValue::SizeTy>(ScalarSpecConst::Size));
    uint32_t Got = 0;
    std::memcpy(&Got, DVPropIt->second.asByteArray(), sizeof(Got));
    EXPECT_EQ(Got, ScalarSpecConst::DefaultValue);
  }
  EXPECT_TRUE(Found) << "Expected my_spec_const in serialized SYCLBIN";
}
