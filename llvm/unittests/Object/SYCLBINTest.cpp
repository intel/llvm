#include "llvm/Object/SYCLBIN.h"
#include "llvm/SYCLPostLink/ModuleSplitter.h"
#include "llvm/Support/FileOutputBuffer.h"

#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <random>

using namespace llvm;
using namespace llvm::sys;
using namespace llvm::object;
using namespace llvm::util;

#define ASSERT_NO_ERROR(x)                                                     \
  if (std::error_code ASSERT_NO_ERROR_ec = x) {                                \
    SmallString<128> MessageStorage;                                           \
    raw_svector_ostream Message(MessageStorage);                               \
    Message << #x ": did not return errc::success.\n"                          \
            << "error number: " << ASSERT_NO_ERROR_ec.value() << "\n"          \
            << "error message: " << ASSERT_NO_ERROR_ec.message() << "\n";      \
    GTEST_FATAL_FAILURE_(MessageStorage.c_str());                              \
  } else {                                                                     \
  }

namespace {
// Helper class for ensuring the destruction of
class ManagedBinaryFile {
public:
  ManagedBinaryFile(ManagedBinaryFile &&) = default;
  ManagedBinaryFile(const ManagedBinaryFile &) = delete;

  ManagedBinaryFile &operator=(ManagedBinaryFile &&) = default;
  ManagedBinaryFile &operator=(const ManagedBinaryFile &) = delete;

  ~ManagedBinaryFile() { fs::remove(FileName); }

  static Expected<ManagedBinaryFile> create(StringRef FileName,
                                            const std::vector<uint8_t> &Data) {
    Expected<std::unique_ptr<FileOutputBuffer>> OutputOrErr =
        FileOutputBuffer::create(FileName, Data.size());
    if (!OutputOrErr)
      return OutputOrErr.takeError();
    ManagedBinaryFile ManagedFile{FileName};
    std::unique_ptr<FileOutputBuffer> Output = std::move(*OutputOrErr);
    llvm::copy(Data, Output->getBufferStart());
    if (Error E = Output->commit())
      return E;

    return std::move(ManagedFile);
  }

private:
  ManagedBinaryFile(StringRef FileName) : FileName{FileName.str()} {}

  std::string FileName;
};

std::vector<uint8_t> generateRandomImage() {
  static std::mt19937 Rng(42);
  static std::uniform_int_distribution<uint64_t> SizeDist(0, 256);
  static std::uniform_int_distribution<uint16_t> BinaryDist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
  std::vector<uint8_t> Image(SizeDist(Rng));
  std::generate(Image.begin(), Image.end(), [&]() { return BinaryDist(Rng); });
  return Image;
}

std::vector<std::vector<uint8_t>> generateUniqueRandomImages(size_t NumImages) {
  std::vector<std::vector<uint8_t>> Images;
  Images.reserve(NumImages);
  Images.emplace_back(generateRandomImage());
  for (size_t I = 1; I < NumImages; ++I) {
    std::vector<uint8_t> GenImage;
    auto CheckImageExist = [&](const std::vector<uint8_t> &Image) {
      return GenImage.size() == Image.size() &&
             std::memcmp(GenImage.data(), Image.data(), GenImage.size()) == 0;
    };
    // Generate images until we get a unique one.
    do {
      GenImage = generateRandomImage();
    } while (std::any_of(Images.begin(), Images.end(), CheckImageExist));
    Images.emplace_back(std::move(GenImage));
  }
  return Images;
}

template <size_t NumIRModules, size_t NumNativeDeviceCodeImages>
void CommonCheck() {
  constexpr size_t NumImages = NumIRModules + NumNativeDeviceCodeImages;
  constexpr SYCLBIN::BundleState State = SYCLBIN::BundleState::Input;
  static constexpr char Arch[] = "some-arch";
  static constexpr char IRMTarget[] = "spir64-unknown-unknown";
  static constexpr char NDCITarget[] = "spir64_gen-unknown-unknown";

  // Create unique temporary directory for these tests
  SmallString<128> TestDirectory;
  {
    ASSERT_NO_ERROR(
        fs::createUniqueDirectory("SYCLBINTest-test", TestDirectory));
  }

  // Create random files to be used in the SYCLBIN. If the same image is
  // generated, retry until they are unique.
  std::vector<std::vector<uint8_t>> Images =
      generateUniqueRandomImages(NumImages);

  std::vector<ManagedBinaryFile> ManagedModuleFiles;
  SmallVector<SYCLBIN::SYCLBINModuleDesc> MDs;
  ManagedModuleFiles.reserve(NumImages);
  MDs.reserve(NumImages);
  for (size_t I = 0; I < NumImages; ++I) {
    SmallString<128> File(TestDirectory);
    llvm::sys::path::append(File, "image_file" + std::to_string(I));
    Expected<ManagedBinaryFile> ManagedModuleFileOrError =
        ManagedBinaryFile::create(File, Images[I]);
    ASSERT_THAT_EXPECTED(ManagedModuleFileOrError, Succeeded());
    ManagedModuleFiles.emplace_back(std::move(*ManagedModuleFileOrError));
    std::vector<module_split::SplitModule> SplitModules{
        module_split::SplitModule{File.c_str(), util::PropertySetRegistry{},
                                  ""}};
    const char *ArchStr = I < NumIRModules ? "" : Arch;
    Triple Target{I < NumIRModules ? IRMTarget : NDCITarget};
    MDs.emplace_back(
        SYCLBIN::SYCLBINModuleDesc{ArchStr, Target, std::move(SplitModules)});
  }

  // Create IR modules.
  SYCLBIN::SYCLBINDesc Desc{State, MDs};

  // Serialize the SYCLBIN.
  SmallString<0> SYCLBINImage;
  raw_svector_ostream SYCLBINImageOS{SYCLBINImage};
  if (Error E = SYCLBIN::write(Desc, SYCLBINImageOS))
    FAIL() << "Failed to write SYCLBIN.";

  // Deserialize the SYCLBIN.
  std::unique_ptr<MemoryBuffer> SYCBINDataBuffer =
      MemoryBuffer::getMemBuffer(SYCLBINImage, /*BufferName=*/"",
                                 /*RequiresNullTerminator=*/false);
  Expected<std::unique_ptr<SYCLBIN>> SYCLBINObjOrError =
      SYCLBIN::read(*SYCBINDataBuffer);
  ASSERT_THAT_EXPECTED(SYCLBINObjOrError, Succeeded());
  SYCLBIN *SYCLBINObj = SYCLBINObjOrError->get();
  ASSERT_NE(SYCLBINObj, nullptr);

  EXPECT_EQ(SYCLBINObj->Version, SYCLBIN::CurrentVersion);

  ASSERT_NE(SYCLBINObj->GlobalMetadata.get(), nullptr);
  SmallString<16> GlobalMetadataKey{
      PropertySetRegistry::SYCLBIN_GLOBAL_METADATA};
  const auto &GlobalMetadataIt =
      SYCLBINObj->GlobalMetadata->getPropSets().find(GlobalMetadataKey);
  ASSERT_NE(GlobalMetadataIt, SYCLBINObj->GlobalMetadata->end());
  const PropertySet &GlobalMetadata = GlobalMetadataIt->second;
  EXPECT_EQ(GlobalMetadata.size(), size_t{1});

  SmallString<16> GlobalMetadataStateKey{"state"};
  const auto &GlobalMetadataStateIt =
      GlobalMetadata.find(GlobalMetadataStateKey);
  ASSERT_NE(GlobalMetadataStateIt, GlobalMetadata.end());
  const PropertyValue &GlobalMetadataState = GlobalMetadataStateIt->second;
  ASSERT_EQ(GlobalMetadataState.getType(), PropertyValue::Type::UINT32);
  EXPECT_EQ(GlobalMetadataState.asUint32(), static_cast<uint32_t>(State));

  // Currently we have an abstract module per image.
  ASSERT_EQ(SYCLBINObj->AbstractModules.size(), size_t{NumImages});

  std::vector<decltype(Images)::const_iterator> ImageIts;
  ImageIts.reserve(NumImages);
  size_t ObservedIRModules = 0, ObservedNativeDeviceCodeImages = 0;
  for (size_t I = 0; I < NumImages; ++I) {
    const SYCLBIN::AbstractModule &AM = SYCLBINObj->AbstractModules[I];

    // This metadata should be the same as in the corresponding split module, so
    // testing should be expanded to ensure preservation.
    ASSERT_NE(AM.Metadata.get(), nullptr);
    EXPECT_TRUE(AM.Metadata->getPropSets().empty());

    ObservedIRModules += AM.IRModules.size();
    for (size_t J = 0; J < AM.IRModules.size(); ++J) {
      const SYCLBIN::IRModule &IRM = AM.IRModules[J];
      ASSERT_NE(IRM.Metadata.get(), nullptr);
      SmallString<16> IRMMetadataKey{
          PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA};
      const auto &IRMMetadataIt =
          IRM.Metadata->getPropSets().find(IRMMetadataKey);
      ASSERT_NE(IRMMetadataIt, IRM.Metadata->end());
      const PropertySet &IRMMetadata = IRMMetadataIt->second;
      EXPECT_EQ(IRMMetadata.size(), size_t{2});

      // The type is currently locked to SPIR-V. This will change in the future.
      SmallString<16> IRMMetadataTypeKey{"type"};
      const auto &IRMMetadataTypeIt = IRMMetadata.find(IRMMetadataTypeKey);
      ASSERT_NE(IRMMetadataTypeIt, IRMMetadata.end());
      const PropertyValue &IRMMetadataType = IRMMetadataTypeIt->second;
      ASSERT_EQ(IRMMetadataType.getType(), PropertyValue::Type::UINT32);
      EXPECT_EQ(IRMMetadataType.asUint32(), uint32_t{0});

      // Make sure the triple string is preserved.
      SmallString<16> IRMMetadataTargetKey{"target"};
      const auto &IRMMetadataTargetIt = IRMMetadata.find(IRMMetadataTargetKey);
      ASSERT_NE(IRMMetadataTargetIt, IRMMetadata.end());
      const PropertyValue &IRMMetadataTarget = IRMMetadataTargetIt->second;
      ASSERT_EQ(IRMMetadataTarget.getType(), PropertyValue::Type::BYTE_ARRAY);
      ASSERT_EQ(IRMMetadataTarget.getByteArraySize(), strlen(IRMTarget));
      EXPECT_EQ(std::memcmp(IRMMetadataTarget.asByteArray(), IRMTarget,
                            strlen(IRMTarget)),
                0);

      // Find the image that matches.
      std::vector<uint8_t> IRImage{IRM.RawIRBytes.begin(),
                                   IRM.RawIRBytes.end()};
      auto ImageMatchIt = std::find(Images.begin(), Images.end(), IRImage);
      ASSERT_NE(ImageMatchIt, Images.end());
      ImageIts.push_back(ImageMatchIt);
    }

    ObservedNativeDeviceCodeImages += AM.NativeDeviceCodeImages.size();
    for (size_t J = 0; J < AM.NativeDeviceCodeImages.size(); ++J) {
      const SYCLBIN::NativeDeviceCodeImage &NDCI = AM.NativeDeviceCodeImages[J];

      ASSERT_NE(NDCI.Metadata.get(), nullptr);
      SmallString<16> NDCIMetadataKey{
          PropertySetRegistry::SYCLBIN_NATIVE_DEVICE_CODE_IMAGE_METADATA};
      const auto &NDCIMetadataIt =
          NDCI.Metadata->getPropSets().find(NDCIMetadataKey);
      ASSERT_NE(NDCIMetadataIt, NDCI.Metadata->end());
      const PropertySet &NDCIMetadata = NDCIMetadataIt->second;
      ASSERT_EQ(NDCIMetadata.size(), size_t{2});

      // Make sure the arch string is preserved.
      SmallString<16> NDCIMetadataArchKey{"arch"};
      const auto &NDCIMetadataArchIt = NDCIMetadata.find(NDCIMetadataArchKey);
      ASSERT_NE(NDCIMetadataArchIt, NDCIMetadata.end());
      const PropertyValue &NDCIMetadataArch = NDCIMetadataArchIt->second;
      ASSERT_EQ(NDCIMetadataArch.getType(), PropertyValue::Type::BYTE_ARRAY);
      ASSERT_EQ(NDCIMetadataArch.getByteArraySize(), strlen(Arch));
      EXPECT_EQ(std::memcmp(NDCIMetadataArch.asByteArray(), Arch, strlen(Arch)),
                0);

      // Make sure the triple string is preserved.
      SmallString<16> NDCIMetadataTargetKey{"target"};
      const auto &NDCIMetadataTargetIt =
          NDCIMetadata.find(NDCIMetadataTargetKey);
      ASSERT_NE(NDCIMetadataTargetIt, NDCIMetadata.end());
      const PropertyValue &NDCIMetadataTarget = NDCIMetadataTargetIt->second;
      ASSERT_EQ(NDCIMetadataTarget.getType(), PropertyValue::Type::BYTE_ARRAY);
      ASSERT_EQ(NDCIMetadataTarget.getByteArraySize(), strlen(NDCITarget));
      EXPECT_EQ(std::memcmp(NDCIMetadataTarget.asByteArray(), NDCITarget,
                            strlen(NDCITarget)),
                0);

      // Find the image that matches.
      std::vector<uint8_t> RawDeviceCodeImage{
          NDCI.RawDeviceCodeImageBytes.begin(),
          NDCI.RawDeviceCodeImageBytes.end()};
      auto ImageMatchIt =
          std::find(Images.begin(), Images.end(), RawDeviceCodeImage);
      ASSERT_NE(ImageMatchIt, Images.end());
      ImageIts.push_back(ImageMatchIt);
    }
  }
  ASSERT_EQ(NumIRModules, ObservedIRModules);
  ASSERT_EQ(NumNativeDeviceCodeImages, ObservedNativeDeviceCodeImages);
  // The images may not appear in the same order they were written. They
  // shouldn't be the same however.
  ASSERT_TRUE(std::unique(ImageIts.begin(), ImageIts.end()) == ImageIts.end());
}

} // namespace

TEST(SYCLBINTest, checkSYCLBINBinaryBasicIRModule) {
  CommonCheck</*NumIRModules=*/1, /*NumNativeDeviceCodeImages=*/0>();
}

TEST(SYCLBINTest, checkSYCLBINBinaryDoubleBasicIRModules) {
  CommonCheck</*NumIRModules=*/2, /*NumNativeDeviceCodeImages=*/0>();
}

TEST(SYCLBINTest, checkSYCLBINBinaryBasicNativeDeviceCodeImage) {
  CommonCheck</*NumIRModules=*/0, /*NumNativeDeviceCodeImages=*/1>();
}

TEST(SYCLBINTest, checkSYCLBINBinaryDoubleBasicNativeDeviceCodeImages) {
  CommonCheck</*NumIRModules=*/0, /*NumNativeDeviceCodeImages=*/2>();
}

TEST(SYCLBINTest, checkSYCLBINBinaryMixBasicImages) {
  CommonCheck</*NumIRModules=*/1, /*NumNativeDeviceCodeImages=*/1>();
}

namespace {

// Build a minimal valid v1 (legacy) SYCLBIN buffer matching the format
// produced by pre-v2 toolchains:
//   [FileHeader: SYBI magic, version=1]
//   [AbstractModuleHeader] [IRModuleHeader]
//   [metadata table: global metadata, AM metadata, IR metadata]
//   [binary table: raw IR bytes]
// All sections aligned to 8.
std::vector<uint8_t> buildLegacyV1Buffer() {
  // Serialize PropertySetRegistry blobs.
  auto serialize = [](const PropertySetRegistry &Reg) {
    SmallString<0> S;
    raw_svector_ostream OS(S);
    Reg.write(OS);
    return std::vector<uint8_t>(S.begin(), S.end());
  };

  PropertySetRegistry Global;
  Global.add(PropertySetRegistry::SYCLBIN_GLOBAL_METADATA, "state",
             static_cast<uint32_t>(SYCLBIN::BundleState::Input));
  std::vector<uint8_t> GlobalMD = serialize(Global);

  PropertySetRegistry AMReg;
  std::vector<uint8_t> AMMD = serialize(AMReg);

  PropertySetRegistry IRReg;
  IRReg.add(PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA, "type",
            uint32_t{0});
  IRReg.add(PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA, "target",
            StringRef("spir64-unknown-unknown"));
  std::vector<uint8_t> IRMD = serialize(IRReg);

  std::vector<uint8_t> RawIR{0xDE, 0xAD, 0xBE, 0xEF};

  // Magic + version + AM count(1) + IR count(1) + NDCI count(0) + ...
  struct alignas(8) FileHeader {
    uint32_t Magic;
    uint32_t Version;
    uint32_t AbstractModuleCount;
    uint32_t IRModuleCount;
    uint32_t NativeDeviceCodeImageCount;
    uint64_t MetadataByteTableSize;
    uint64_t BinaryByteTableSize;
    uint64_t GlobalMetadataOffset;
    uint64_t GlobalMetadataSize;
  };
  struct alignas(8) AbstractModuleHeader {
    uint64_t MetadataOffset;
    uint64_t MetadataSize;
    uint32_t IRModuleCount;
    uint32_t IRModuleOffset;
    uint32_t NativeDeviceCodeImageCount;
    uint32_t NativeDeviceCodeImageOffset;
  };
  struct alignas(8) IRModuleHeader {
    uint64_t MetadataOffset;
    uint64_t MetadataSize;
    uint64_t RawIRBytesOffset;
    uint64_t RawIRBytesSize;
  };

  // Metadata table layout (in v1 read order): global, AM, IR.
  uint64_t GlobalOff = 0;
  uint64_t AMOff = GlobalOff + GlobalMD.size();
  uint64_t IROff = AMOff + AMMD.size();
  uint64_t MetadataTotal = IROff + IRMD.size();
  uint64_t MetadataAligned = (MetadataTotal + 7u) & ~uint64_t{7};

  std::vector<uint8_t> MetadataTable(MetadataAligned, 0);
  std::memcpy(&MetadataTable[GlobalOff], GlobalMD.data(), GlobalMD.size());
  std::memcpy(&MetadataTable[AMOff], AMMD.data(), AMMD.size());
  std::memcpy(&MetadataTable[IROff], IRMD.data(), IRMD.size());

  uint64_t BinaryTotal = RawIR.size();
  std::vector<uint8_t> BinaryTable(BinaryTotal);
  std::memcpy(BinaryTable.data(), RawIR.data(), RawIR.size());

  FileHeader FH{};
  FH.Magic = SYCLBIN::LegacyMagicNumber;
  FH.Version = 1;
  FH.AbstractModuleCount = 1;
  FH.IRModuleCount = 1;
  FH.NativeDeviceCodeImageCount = 0;
  FH.MetadataByteTableSize = MetadataTotal;
  FH.BinaryByteTableSize = BinaryTotal;
  FH.GlobalMetadataOffset = GlobalOff;
  FH.GlobalMetadataSize = GlobalMD.size();

  AbstractModuleHeader AMH{};
  AMH.MetadataOffset = AMOff;
  AMH.MetadataSize = AMMD.size();
  AMH.IRModuleCount = 1;
  AMH.IRModuleOffset = 0;
  AMH.NativeDeviceCodeImageCount = 0;
  AMH.NativeDeviceCodeImageOffset = 0;

  IRModuleHeader IRH{};
  IRH.MetadataOffset = IROff;
  IRH.MetadataSize = IRMD.size();
  IRH.RawIRBytesOffset = 0;
  IRH.RawIRBytesSize = RawIR.size();

  std::vector<uint8_t> Buf;
  auto append = [&](const void *Ptr, size_t Size) {
    const uint8_t *Bytes = static_cast<const uint8_t *>(Ptr);
    Buf.insert(Buf.end(), Bytes, Bytes + Size);
  };
  append(&FH, sizeof(FH));
  append(&AMH, sizeof(AMH));
  append(&IRH, sizeof(IRH));
  Buf.insert(Buf.end(), MetadataTable.begin(), MetadataTable.end());
  Buf.insert(Buf.end(), BinaryTable.begin(), BinaryTable.end());
  return Buf;
}

} // namespace

// Backward-compat regression: SYCLBIN::read must continue to accept the v1
// (legacy SYBI-magic) on-disk format produced by pre-v2 toolchains.
TEST(SYCLBINTest, checkLegacyV1ReaderBackwardCompat) {
  std::vector<uint8_t> Buf = buildLegacyV1Buffer();
  std::unique_ptr<MemoryBuffer> MemBuf = MemoryBuffer::getMemBufferCopy(
      StringRef(reinterpret_cast<const char *>(Buf.data()), Buf.size()));

  Expected<std::unique_ptr<SYCLBIN>> Result = SYCLBIN::read(*MemBuf);
  ASSERT_THAT_EXPECTED(Result, Succeeded());
  std::unique_ptr<SYCLBIN> Obj = std::move(*Result);
  ASSERT_NE(Obj.get(), nullptr);

  EXPECT_EQ(Obj->Version, uint32_t{1});

  ASSERT_NE(Obj->GlobalMetadata.get(), nullptr);
  SmallString<16> GMKey{PropertySetRegistry::SYCLBIN_GLOBAL_METADATA};
  auto It = Obj->GlobalMetadata->getPropSets().find(GMKey);
  ASSERT_NE(It, Obj->GlobalMetadata->end());
  auto StateIt = It->second.find(SmallString<16>{"state"});
  ASSERT_NE(StateIt, It->second.end());
  EXPECT_EQ(StateIt->second.asUint32(),
            static_cast<uint32_t>(SYCLBIN::BundleState::Input));

  ASSERT_EQ(Obj->AbstractModules.size(), size_t{1});
  const SYCLBIN::AbstractModule &AM = Obj->AbstractModules[0];
  ASSERT_EQ(AM.IRModules.size(), size_t{1});
  ASSERT_EQ(AM.NativeDeviceCodeImages.size(), size_t{0});

  const SYCLBIN::IRModule &IRM = AM.IRModules[0];
  ASSERT_EQ(IRM.RawIRBytes.size(), size_t{4});
  EXPECT_EQ(static_cast<uint8_t>(IRM.RawIRBytes[0]), 0xDE);
  EXPECT_EQ(static_cast<uint8_t>(IRM.RawIRBytes[1]), 0xAD);
  EXPECT_EQ(static_cast<uint8_t>(IRM.RawIRBytes[2]), 0xBE);
  EXPECT_EQ(static_cast<uint8_t>(IRM.RawIRBytes[3]), 0xEF);
}
