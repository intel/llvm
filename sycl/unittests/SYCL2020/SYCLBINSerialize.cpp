//==----------- SYCLBINSerialize.cpp - SYCLBIN serialization tests ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/kernel_bundle_impl.hpp>
#include <detail/syclbin.hpp>
#include <sycl/sycl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <optional>

class SYCLBINSerializeKernel;
MOCK_INTEGRATION_HEADER(SYCLBINSerializeKernel)

namespace {

// Builds a single AOT-style mock device image carrying an explicit
// "compile_target" property in its [SYCL/device requirements] property set.
// The kernel_bundle ext_oneapi_get_content path must forward this property
// verbatim into the resulting SYCLBIN's per-image metadata so that re-loading
// the SYCLBIN can recover the same compile_target.
sycl::unittest::MockDeviceImage
makeImageWithCompileTarget(const std::string &KernelName,
                           const std::string &CompileTarget) {
  using namespace sycl::unittest;

  // Pack the compile_target string as a base64-encodable byte array, with the
  // 8-byte size prefix that the SYCL runtime expects for byte-array
  // properties.
  std::vector<char> Data(8 + CompileTarget.size());
  std::copy(CompileTarget.begin(), CompileTarget.end(), Data.data() + 8);
  MockProperty CompileTargetProp("compile_target", std::move(Data),
                                 SYCL_PROPERTY_TYPE_BYTE_ARRAY);

  MockPropertySet PropSet;
  PropSet.insert(__SYCL_PROPERTY_SET_SYCL_DEVICE_REQUIREMENTS,
                 std::move(CompileTargetProp));

  std::vector<unsigned char> Bin{0xDE, 0xAD, 0xBE, 0xEF};
  std::vector<MockOffloadEntry> Entries = makeEmptyKernels({KernelName});

  return MockDeviceImage{SYCL_DEVICE_BINARY_TYPE_NATIVE,
                         __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN,
                         /*CompileOptions=*/"",
                         /*LinkOptions=*/"",
                         std::move(Bin),
                         std::move(Entries),
                         std::move(PropSet)};
}

sycl::unittest::MockDeviceImage Imgs[] = {
    makeImageWithCompileTarget("SYCLBINSerializeKernel", "intel_gpu_pvc")};
sycl::unittest::MockDeviceImageArray<std::size(Imgs)> ImgArray{Imgs};

// Strips the OffloadBinary wrapper that ext_oneapi_get_content emits and
// returns a string_view over the contained raw SYCLBIN bytes. Returns
// std::nullopt if the wrapper does not parse cleanly; callers should ASSERT
// on the optional rather than dereferencing unconditionally.
std::optional<std::string_view>
stripOffloadWrapper(const std::vector<char> &Bytes) {
  // OffloadBinary v2 layout: 32-byte header, then a single entry pointing at
  // the SYCLBIN payload. See sycl/source/detail/syclbin.cpp for layout.
  struct OffloadBinaryHeader {
    uint8_t Magic[4];
    uint32_t Version;
    uint64_t Size;
    uint64_t EntriesOffset;
    uint64_t EntriesCount;
  };
  struct OffloadBinaryEntry {
    uint16_t ImageKind;
    uint16_t OffloadKind;
    uint32_t Flags;
    uint64_t StringOffset;
    uint64_t NumStrings;
    uint64_t ImageOffset;
    uint64_t ImageSize;
  };

  if (Bytes.size() < sizeof(OffloadBinaryHeader))
    return std::nullopt;
  const auto *H = reinterpret_cast<const OffloadBinaryHeader *>(Bytes.data());
  if (H->Magic[0] != 0x10 || H->Magic[1] != 0xFF || H->Magic[2] != 0x10 ||
      H->Magic[3] != 0xAD)
    return std::nullopt;
  if (H->Version != 2u || H->EntriesCount != 1u)
    return std::nullopt;
  if (H->EntriesOffset + sizeof(OffloadBinaryEntry) > Bytes.size())
    return std::nullopt;

  const auto *E = reinterpret_cast<const OffloadBinaryEntry *>(
      Bytes.data() + H->EntriesOffset);
  if (E->ImageKind != /*IMG_SYCLBIN*/ 7)
    return std::nullopt;
  if (E->ImageOffset + E->ImageSize > Bytes.size())
    return std::nullopt;
  return std::string_view{Bytes.data() + E->ImageOffset,
                          static_cast<size_t>(E->ImageSize)};
}

} // namespace

// Round-trip test: emit a SYCLBIN from a kernel_bundle whose underlying device
// image carries a compile_target property, then parse the SYCLBIN back and
// confirm the property survived.
TEST(SYCLBINSerialize, CompileTargetRoundTrip) {
  sycl::unittest::UrMock<> Mock;

  const sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};

  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev});

  std::vector<char> Bytes =
      sycl::detail::getSyclObjImpl(KB)->ext_oneapi_get_content();

  ASSERT_FALSE(Bytes.empty());

  // The runtime SYCLBIN ctor expects the OffloadBinary wrapper, so feed the
  // full byte stream.
  sycl::detail::SYCLBIN Parsed{Bytes.data(), Bytes.size()};

  // Find the "compile_target" property in any abstract module's metadata and
  // assert it round-tripped to "intel_gpu_pvc".
  bool Found = false;
  for (const auto &AM : Parsed.AbstractModules) {
    ASSERT_NE(AM.Metadata, nullptr);
    auto It = AM.Metadata->getPropSets().find(
        __SYCL_PROPERTY_SET_SYCL_DEVICE_REQUIREMENTS);
    if (It == AM.Metadata->getPropSets().end())
      continue;
    auto PIt = It->second.find("compile_target");
    if (PIt == It->second.end())
      continue;
    const sycl::detail::PropertyValue &Val = PIt->second;
    ASSERT_EQ(Val.getType(), sycl::detail::PropertyValue::BYTE_ARRAY);
    std::string Got{reinterpret_cast<const char *>(Val.asByteArray()),
                    static_cast<size_t>(Val.getByteArraySize())};
    EXPECT_EQ(Got, "intel_gpu_pvc");
    Found = true;
  }
  EXPECT_TRUE(Found) << "compile_target property was lost in serialization";
}

// Sanity test: the bundle_state global metadata round-trips correctly across
// every state allowed by the spec (everything except ext_oneapi_source).
namespace {
template <sycl::bundle_state State>
void runBundleStateRoundTripAt(const sycl::context &Ctx,
                               const sycl::device &Dev) {
  auto KB = sycl::get_kernel_bundle<State>(Ctx, {Dev});

  std::vector<char> Bytes =
      sycl::detail::getSyclObjImpl(KB)->ext_oneapi_get_content();
  ASSERT_FALSE(Bytes.empty());

  sycl::detail::SYCLBIN Parsed{Bytes.data(), Bytes.size()};
  ASSERT_NE(Parsed.GlobalMetadata, nullptr);
  auto It = Parsed.GlobalMetadata->getPropSets().find(
      sycl::detail::PropertySetRegistry::SYCLBIN_GLOBAL_METADATA);
  ASSERT_NE(It, Parsed.GlobalMetadata->getPropSets().end());
  auto PIt = It->second.find("state");
  ASSERT_NE(PIt, It->second.end());
  EXPECT_EQ(PIt->second.asUint32(), static_cast<uint32_t>(State));
}
} // namespace

TEST(SYCLBINSerialize, BundleStateRoundTrip) {
  sycl::unittest::UrMock<> Mock;
  const sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};
  runBundleStateRoundTripAt<sycl::bundle_state::executable>(Ctx, Dev);
}

TEST(SYCLBINSerialize, BundleStateRoundTripInput) {
  sycl::unittest::UrMock<> Mock;
  const sycl::device Dev = sycl::platform().get_devices()[0];
  if (!Dev.has(sycl::aspect::online_compiler))
    GTEST_SKIP() << "Device lacks aspect::online_compiler.";
  sycl::context Ctx{Dev};
  runBundleStateRoundTripAt<sycl::bundle_state::input>(Ctx, Dev);
}

TEST(SYCLBINSerialize, BundleStateRoundTripObject) {
  sycl::unittest::UrMock<> Mock;
  const sycl::device Dev = sycl::platform().get_devices()[0];
  if (!Dev.has(sycl::aspect::online_linker))
    GTEST_SKIP() << "Device lacks aspect::online_linker.";
  sycl::context Ctx{Dev};
  // Construct an object-state bundle via compile() of an input-state bundle,
  // matching how the public API exposes object-state kernel_bundles.
  auto Input = sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});
  auto Obj = sycl::compile(Input);

  std::vector<char> Bytes =
      sycl::detail::getSyclObjImpl(Obj)->ext_oneapi_get_content();
  ASSERT_FALSE(Bytes.empty());

  sycl::detail::SYCLBIN Parsed{Bytes.data(), Bytes.size()};
  ASSERT_NE(Parsed.GlobalMetadata, nullptr);
  auto It = Parsed.GlobalMetadata->getPropSets().find(
      sycl::detail::PropertySetRegistry::SYCLBIN_GLOBAL_METADATA);
  ASSERT_NE(It, Parsed.GlobalMetadata->getPropSets().end());
  auto PIt = It->second.find("state");
  ASSERT_NE(PIt, It->second.end());
  EXPECT_EQ(PIt->second.asUint32(),
            static_cast<uint32_t>(sycl::bundle_state::object));
}

// The image payload bytes must round-trip through the binary byte table
// unchanged. This guards against a header-vs-table layout drift that header
// inspection alone (e.g. CompileTargetRoundTrip) does not catch.
TEST(SYCLBINSerialize, ImageBytesRoundTrip) {
  sycl::unittest::UrMock<> Mock;

  const sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};

  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev});

  std::vector<char> Bytes =
      sycl::detail::getSyclObjImpl(KB)->ext_oneapi_get_content();

  sycl::detail::SYCLBIN Parsed{Bytes.data(), Bytes.size()};

  // Mock image payload set in makeImageWithCompileTarget above is the 4-byte
  // sequence {0xDE, 0xAD, 0xBE, 0xEF}. We emit a single native module per
  // image, so expect to see exactly that payload come back.
  bool FoundPayload = false;
  for (const auto &AM : Parsed.AbstractModules) {
    for (const auto &NDCI : AM.NativeDeviceCodeImages) {
      if (NDCI.RawDeviceCodeImageBytes.size() != 4)
        continue;
      const auto *P = reinterpret_cast<const unsigned char *>(
          NDCI.RawDeviceCodeImageBytes.data());
      if (P[0] == 0xDE && P[1] == 0xAD && P[2] == 0xBE && P[3] == 0xEF)
        FoundPayload = true;
    }
  }
  EXPECT_TRUE(FoundPayload)
      << "Native image payload bytes did not round-trip correctly.";
}

// An empty SYCLBINDesc (no abstract modules) must still produce a valid
// OffloadBinary-wrapped SYCLBIN whose header is parseable and whose abstract
// module count is zero. Exercises the boundary case of serializeImages with
// zero device images, without needing a kernel_bundle at all.
TEST(SYCLBINSerialize, EmptyBundle) {
  // Synthesize the smallest valid SYCLBINDesc: only the global state metadata,
  // no images.
  sycl::detail::SYCLBIN::SYCLBINDesc Desc;
  sycl::detail::PropertySetRegistry GlobalProps;
  GlobalProps.add(sycl::detail::PropertySetRegistry::SYCLBIN_GLOBAL_METADATA,
                  "state",
                  static_cast<uint32_t>(sycl::bundle_state::executable));
  std::ostringstream GMOS;
  GlobalProps.write(GMOS);
  Desc.GlobalMetadata = GMOS.str();

  std::vector<char> Bytes = sycl::detail::SYCLBIN::write(Desc);
  ASSERT_FALSE(Bytes.empty());

  // Reader must accept zero-AM SYCLBINs without throwing.
  sycl::detail::SYCLBIN Parsed{Bytes.data(), Bytes.size()};
  EXPECT_EQ(Parsed.AbstractModules.size(), 0u);
  EXPECT_EQ(Parsed.Version, 1u);
  ASSERT_NE(Parsed.GlobalMetadata, nullptr);
  auto It = Parsed.GlobalMetadata->getPropSets().find(
      sycl::detail::PropertySetRegistry::SYCLBIN_GLOBAL_METADATA);
  ASSERT_NE(It, Parsed.GlobalMetadata->getPropSets().end());
  EXPECT_EQ(It->second.find("state")->second.asUint32(),
            static_cast<uint32_t>(sycl::bundle_state::executable));
}

// Specialization constant property sets must round-trip byte-for-byte through
// SYCLBIN serialization. Builds a minimal abstract module whose metadata
// carries a synthetic [SYCL/specialization constants] entry, writes it, parses
// it back, and asserts the property byte payload is unchanged.
TEST(SYCLBINSerialize, SpecConstantsRoundTrip) {
  using namespace sycl::detail;

  // Synthetic spec const descriptor: { ID=42, Offset=0, Size=4 }, packed as
  // three little-endian uint32_t. PropertyValue stores this as a BYTE_ARRAY.
  const std::array<uint32_t, 3> SpecConstDescr{42u, 0u, 4u};
  const PropertyValue::byte *DescPtr =
      reinterpret_cast<const PropertyValue::byte *>(SpecConstDescr.data());
  const PropertyValue::SizeTy DescSizeBits =
      sizeof(SpecConstDescr) * 8; // 12 bytes -> 96 bits.
  PropertyValue SpecConstValue{DescPtr, DescSizeBits};

  // Build per-AM metadata containing the [SYCL/specialization constants]
  // property set with our synthetic entry.
  PropertySetRegistry AMProps;
  PropertySet &SCSet = AMProps["SYCL/specialization constants"];
  SCSet["my_spec_const"] = std::move(SpecConstValue);
  std::ostringstream AMOS;
  AMProps.write(AMOS);
  std::string AMMetadata = AMOS.str();

  // Wrap into an empty IR module so the abstract module is otherwise minimal.
  SYCLBIN::SYCLBINDesc Desc;
  PropertySetRegistry GlobalProps;
  GlobalProps.add(PropertySetRegistry::SYCLBIN_GLOBAL_METADATA, "state",
                  static_cast<uint32_t>(sycl::bundle_state::executable));
  std::ostringstream GMOS;
  GlobalProps.write(GMOS);
  Desc.GlobalMetadata = GMOS.str();

  std::string IRBytesStorage{"\x01\x02\x03\x04", 4};
  SYCLBIN::AbstractModuleDesc &AM = Desc.AbstractModules.emplace_back();
  AM.Metadata = AMMetadata;
  SYCLBIN::ImageDesc &IRMD = AM.IRModules.emplace_back();
  IRMD.Bytes = IRBytesStorage;
  PropertySetRegistry IRMProps;
  IRMProps.add(PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA, "type", 0u);
  IRMProps.add(PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA, "target",
               std::string_view{"spir64-unknown-unknown"});
  std::ostringstream IRMOS;
  IRMProps.write(IRMOS);
  IRMD.Metadata = IRMOS.str();

  std::vector<char> Bytes = SYCLBIN::write(Desc);
  ASSERT_FALSE(Bytes.empty());

  SYCLBIN Parsed{Bytes.data(), Bytes.size()};
  ASSERT_EQ(Parsed.AbstractModules.size(), 1u);
  ASSERT_NE(Parsed.AbstractModules[0].Metadata, nullptr);

  auto It = Parsed.AbstractModules[0].Metadata->getPropSets().find(
      "SYCL/specialization constants");
  ASSERT_NE(It, Parsed.AbstractModules[0].Metadata->getPropSets().end());
  auto PIt = It->second.find("my_spec_const");
  ASSERT_NE(PIt, It->second.end());

  ASSERT_EQ(PIt->second.getType(), PropertyValue::BYTE_ARRAY);
  EXPECT_EQ(PIt->second.getByteArraySize(), sizeof(SpecConstDescr));
  EXPECT_EQ(std::memcmp(PIt->second.asByteArray(), SpecConstDescr.data(),
                        sizeof(SpecConstDescr)),
            0);
}

// Device-global property sets must round-trip byte-for-byte. The
// [SYCL/device globals] property set carries one byte-array per device global
// containing { uint32_t Size, uint32_t DeviceImageScope } (see
// PropertySets.md). Mirrors the spec const round-trip pattern.
TEST(SYCLBINSerialize, DeviceGlobalsRoundTrip) {
  using namespace sycl::detail;

  // Synthetic device-global descriptor: { Size=128, DeviceImageScope=1 }.
  const std::array<uint32_t, 2> DGDescr{128u, 1u};
  PropertyValue DGValue{
      reinterpret_cast<const PropertyValue::byte *>(DGDescr.data()),
      sizeof(DGDescr) * 8};

  PropertySetRegistry AMProps;
  PropertySet &DGSet = AMProps["SYCL/device globals"];
  DGSet["my_device_global"] = std::move(DGValue);
  std::ostringstream AMOS;
  AMProps.write(AMOS);
  std::string AMMetadata = AMOS.str();

  SYCLBIN::SYCLBINDesc Desc;
  PropertySetRegistry GlobalProps;
  GlobalProps.add(PropertySetRegistry::SYCLBIN_GLOBAL_METADATA, "state",
                  static_cast<uint32_t>(sycl::bundle_state::executable));
  std::ostringstream GMOS;
  GlobalProps.write(GMOS);
  Desc.GlobalMetadata = GMOS.str();

  std::string IRBytesStorage{"\x01\x02\x03\x04", 4};
  SYCLBIN::AbstractModuleDesc &AM = Desc.AbstractModules.emplace_back();
  AM.Metadata = AMMetadata;
  SYCLBIN::ImageDesc &IRMD = AM.IRModules.emplace_back();
  IRMD.Bytes = IRBytesStorage;
  PropertySetRegistry IRMProps;
  IRMProps.add(PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA, "type", 0u);
  IRMProps.add(PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA, "target",
               std::string_view{"spir64-unknown-unknown"});
  std::ostringstream IRMOS;
  IRMProps.write(IRMOS);
  IRMD.Metadata = IRMOS.str();

  std::vector<char> Bytes = SYCLBIN::write(Desc);
  ASSERT_FALSE(Bytes.empty());

  SYCLBIN Parsed{Bytes.data(), Bytes.size()};
  ASSERT_EQ(Parsed.AbstractModules.size(), 1u);
  ASSERT_NE(Parsed.AbstractModules[0].Metadata, nullptr);

  auto It = Parsed.AbstractModules[0].Metadata->getPropSets().find(
      "SYCL/device globals");
  ASSERT_NE(It, Parsed.AbstractModules[0].Metadata->getPropSets().end());
  auto PIt = It->second.find("my_device_global");
  ASSERT_NE(PIt, It->second.end());
  ASSERT_EQ(PIt->second.getType(), PropertyValue::BYTE_ARRAY);
  EXPECT_EQ(PIt->second.getByteArraySize(), sizeof(DGDescr));
  EXPECT_EQ(
      std::memcmp(PIt->second.asByteArray(), DGDescr.data(), sizeof(DGDescr)),
      0);
}

// A SYCLBIN containing both an IR module and a native device code image in
// separate abstract modules must round-trip cleanly: each branch reaches its
// dedicated header array and binary table region. Bypasses kernel_bundle to
// drive SYCLBIN::write directly with a hand-built mix.
TEST(SYCLBINSerialize, MixedIRAndNativeImages) {
  using namespace sycl::detail;

  SYCLBIN::SYCLBINDesc Desc;

  // Global metadata.
  PropertySetRegistry GlobalProps;
  GlobalProps.add(PropertySetRegistry::SYCLBIN_GLOBAL_METADATA, "state",
                  static_cast<uint32_t>(sycl::bundle_state::executable));
  std::ostringstream GMOS;
  GlobalProps.write(GMOS);
  Desc.GlobalMetadata = GMOS.str();

  std::string IRBytesStorage{"\xCA\xFE\xBA\xBE", 4};
  std::string NativeBytesStorage{"\xDE\xAD\xBE\xEF", 4};

  auto serializeProps = [](const PropertySetRegistry &Props) {
    std::ostringstream OS;
    Props.write(OS);
    return OS.str();
  };

  // Abstract module 1: SPIR-V IR module.
  {
    SYCLBIN::AbstractModuleDesc &AM = Desc.AbstractModules.emplace_back();
    AM.Metadata = "";
    SYCLBIN::ImageDesc &IRMD = AM.IRModules.emplace_back();
    IRMD.Bytes = IRBytesStorage;
    PropertySetRegistry IRMProps;
    IRMProps.add(PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA, "type", 0u);
    IRMProps.add(PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA, "target",
                 std::string_view{"spir64-unknown-unknown"});
    IRMD.Metadata = serializeProps(IRMProps);
  }

  // Abstract module 2: native device code image.
  {
    SYCLBIN::AbstractModuleDesc &AM = Desc.AbstractModules.emplace_back();
    AM.Metadata = "";
    SYCLBIN::ImageDesc &NDCID = AM.NativeDeviceCodeImages.emplace_back();
    NDCID.Bytes = NativeBytesStorage;
    PropertySetRegistry NDCIProps;
    NDCIProps.add(
        PropertySetRegistry::SYCLBIN_NATIVE_DEVICE_CODE_IMAGE_METADATA, "arch",
        std::string_view{"intel_gpu_pvc"});
    NDCIProps.add(
        PropertySetRegistry::SYCLBIN_NATIVE_DEVICE_CODE_IMAGE_METADATA,
        "target", std::string_view{"spir64_gen-unknown-unknown"});
    NDCID.Metadata = serializeProps(NDCIProps);
  }

  std::vector<char> Bytes = SYCLBIN::write(Desc);
  ASSERT_FALSE(Bytes.empty());

  SYCLBIN Parsed{Bytes.data(), Bytes.size()};
  ASSERT_EQ(Parsed.AbstractModules.size(), 2u);

  // Cross-check: one AM has only IR, the other has only native, and the byte
  // payloads round-trip unchanged.
  size_t IRPayloadHits = 0;
  size_t NativePayloadHits = 0;
  for (const auto &AM : Parsed.AbstractModules) {
    for (const auto &IRM : AM.IRModules) {
      if (IRM.RawIRBytes == IRBytesStorage)
        ++IRPayloadHits;
    }
    for (const auto &NDCI : AM.NativeDeviceCodeImages) {
      if (NDCI.RawDeviceCodeImageBytes == NativeBytesStorage)
        ++NativePayloadHits;
    }
  }
  EXPECT_EQ(IRPayloadHits, 1u);
  EXPECT_EQ(NativePayloadHits, 1u);
}

// ext_oneapi_get_content of an OffloadBinary-wrapped output must round-trip
// the OffloadBinary header magic, SYCLBIN magic, and SYCLBIN version. Per the
// sycl_ext_oneapi_syclbin test plan section 10.2, the on-disk SYCLBIN version
// must be 1.
TEST(SYCLBINSerialize, OffloadAndSYCLBINMagicsPresent) {
  sycl::unittest::UrMock<> Mock;
  const sycl::device Dev = sycl::platform().get_devices()[0];
  sycl::context Ctx{Dev};

  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev});

  std::vector<char> Bytes =
      sycl::detail::getSyclObjImpl(KB)->ext_oneapi_get_content();

  std::optional<std::string_view> SYCLBINBytesOpt =
      stripOffloadWrapper(Bytes);
  ASSERT_TRUE(SYCLBINBytesOpt.has_value())
      << "OffloadBinary wrapper failed to parse from ext_oneapi_get_content "
         "output.";
  std::string_view SYCLBINBytes = *SYCLBINBytesOpt;

  ASSERT_GE(SYCLBINBytes.size(), 2 * sizeof(uint32_t));
  uint32_t Magic = 0;
  std::memcpy(&Magic, SYCLBINBytes.data(), sizeof(Magic));
  EXPECT_EQ(Magic, sycl::detail::SYCLBIN::MagicNumber);

  uint32_t Version = 0;
  std::memcpy(&Version, SYCLBINBytes.data() + sizeof(uint32_t),
              sizeof(Version));
  EXPECT_EQ(Version, sycl::detail::SYCLBIN::CurrentVersion);
  EXPECT_EQ(Version, 1u);

  // Cross-check via the higher-level parser: it should recover the same
  // version field as the raw bytes carry.
  sycl::detail::SYCLBIN Parsed{Bytes.data(), Bytes.size()};
  EXPECT_EQ(Parsed.Version, 1u);
}
