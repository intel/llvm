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
  std::mt19937 Rng(42);
  std::uniform_int_distribution<uint64_t> SizeDist(0, 256);
  std::uniform_int_distribution<uint16_t> KindDist(0);
  std::uniform_int_distribution<uint16_t> BinaryDist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
  std::vector<uint8_t> Image(SizeDist(Rng));
  std::generate(Image.begin(), Image.end(), [&]() { return BinaryDist(Rng); });
  return Image;
}

} // namespace

TEST(SYCLBINTest, checkSYCLBINBinaryBasicIRModule) {
  constexpr SYCLBIN::BundleState State = SYCLBIN::BundleState::Input;

  // Create unique temporary directory for these tests
  SmallString<128> TestDirectory;
  {
    ASSERT_NO_ERROR(
        fs::createUniqueDirectory("SYCLBINTest-test", TestDirectory));
  }

  // Create a random file to be used in the SYCLBIN.
  SmallString<128> File1(TestDirectory);
  File1.append("/image_file1");
  std::vector<uint8_t> Image = generateRandomImage();
  Expected<ManagedBinaryFile> ManagedModuleFileOrError =
      ManagedBinaryFile::create(File1, Image);
  ASSERT_THAT_EXPECTED(ManagedModuleFileOrError, Succeeded());
  ManagedBinaryFile ManagedModuleFile = std::move(*ManagedModuleFileOrError);

  // Create IR module.
  std::vector<module_split::SplitModule> SplitModules{module_split::SplitModule{
      File1.c_str(), util::PropertySetRegistry{}, ""}};
  SmallVector<SYCLBIN::SYCLBINModuleDesc> MDs{
      SYCLBIN::SYCLBINModuleDesc{"", std::move(SplitModules)}};
  SYCLBIN::SYCLBINDesc Desc{SYCLBIN::BundleState::Input, MDs};
  size_t SYCLBINByteSize = 0;
  if (Error E = Desc.getSYCLBINByteSite().moveInto(SYCLBINByteSize))
    FAIL() << "Failed to get SYCLBIN byte size.";

  // Serialize the SYCLBIN.
  SmallString<0> SYCLBINImage;
  SYCLBINImage.reserve(SYCLBINByteSize);
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

  ASSERT_EQ(SYCLBINObj->AbstractModules.size(), size_t{1});
  const SYCLBIN::AbstractModule &AM = SYCLBINObj->AbstractModules[0];

  // This metadata should be the same as in the corresponding split module, so
  // testing should be expanded to ensure preservation.
  ASSERT_NE(AM.Metadata.get(), nullptr);
  EXPECT_TRUE(AM.Metadata->getPropSets().empty());

  // There was no arch string, so there should be no native device code images.
  EXPECT_TRUE(AM.NativeDeviceCodeImages.empty());

  // There was a single module with no arch string, so we should have an IR
  // module.
  ASSERT_EQ(AM.IRModules.size(), size_t{1});
  const SYCLBIN::IRModule &IRM = AM.IRModules[0];

  ASSERT_NE(IRM.Metadata.get(), nullptr);
  SmallString<16> IRMMetadataKey{
      PropertySetRegistry::SYCLBIN_IR_MODULE_METADATA};
  const auto &IRMMetadataIt = IRM.Metadata->getPropSets().find(IRMMetadataKey);
  ASSERT_NE(IRMMetadataIt, IRM.Metadata->end());
  const PropertySet &IRMMetadata = IRMMetadataIt->second;
  EXPECT_EQ(IRMMetadata.size(), size_t{1});

  // The type is currently locked to SPIR-V. This will change in the future.
  SmallString<16> IRMMetadataTypeKey{"type"};
  const auto &IRMMetadataTypeIt = IRMMetadata.find(IRMMetadataTypeKey);
  ASSERT_NE(IRMMetadataTypeIt, IRMMetadata.end());
  const PropertyValue &IRMMetadataType = IRMMetadataTypeIt->second;
  ASSERT_EQ(IRMMetadataType.getType(), PropertyValue::Type::UINT32);
  EXPECT_EQ(IRMMetadataType.asUint32(), uint32_t{0});

  // Check that the image is the same.
  ASSERT_EQ(Image.size(), IRM.RawIRBytes.size());
  EXPECT_EQ(std::memcmp(Image.data(), IRM.RawIRBytes.data(), Image.size()), 0);
}

TEST(SYCLBINTest, checkSYCLBINBinaryBasicNativeDeviceCodeImage) {
  constexpr SYCLBIN::BundleState State = SYCLBIN::BundleState::Input;
  static constexpr char Arch[] = "some-arch";

  // Create unique temporary directory for these tests
  SmallString<128> TestDirectory;
  {
    ASSERT_NO_ERROR(
        fs::createUniqueDirectory("SYCLBINTest-test", TestDirectory));
  }

  // Create a random file to be used in the SYCLBIN.
  SmallString<128> File1(TestDirectory);
  File1.append("/image_file1");
  std::vector<uint8_t> Image = generateRandomImage();
  Expected<ManagedBinaryFile> ManagedModuleFileOrError =
      ManagedBinaryFile::create(File1, Image);
  ASSERT_THAT_EXPECTED(ManagedModuleFileOrError, Succeeded());
  ManagedBinaryFile ManagedModuleFile = std::move(*ManagedModuleFileOrError);

  // Create IR module.
  std::vector<module_split::SplitModule> SplitModules{module_split::SplitModule{
      File1.c_str(), util::PropertySetRegistry{}, ""}};
  SmallVector<SYCLBIN::SYCLBINModuleDesc> MDs{
      SYCLBIN::SYCLBINModuleDesc{Arch, std::move(SplitModules)}};
  SYCLBIN::SYCLBINDesc Desc{SYCLBIN::BundleState::Input, MDs};
  size_t SYCLBINByteSize = 0;
  if (Error E = Desc.getSYCLBINByteSite().moveInto(SYCLBINByteSize))
    FAIL() << "Failed to get SYCLBIN byte size.";

  // Serialize the SYCLBIN.
  SmallString<0> SYCLBINImage;
  SYCLBINImage.reserve(SYCLBINByteSize);
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

  ASSERT_EQ(SYCLBINObj->AbstractModules.size(), size_t{1});
  const SYCLBIN::AbstractModule &AM = SYCLBINObj->AbstractModules[0];

  // This metadata should be the same as in the corresponding split module, so
  // testing should be expanded to ensure preservation.
  ASSERT_NE(AM.Metadata.get(), nullptr);
  EXPECT_TRUE(AM.Metadata->getPropSets().empty());

  // There was a single module with an arch string, so we should not have an IR
  // module.
  EXPECT_TRUE(AM.IRModules.empty());

  // There was an arch string, so there should be a native device code images.
  ASSERT_EQ(AM.NativeDeviceCodeImages.size(), size_t{1});
  const SYCLBIN::NativeDeviceCodeImage &NDCI = AM.NativeDeviceCodeImages[0];

  ASSERT_NE(NDCI.Metadata.get(), nullptr);
  SmallString<16> NDCIMetadataKey{
      PropertySetRegistry::SYCLBIN_NATIVE_DEVICE_CODE_IMAGE_METADATA};
  const auto &NDCIMetadataIt =
      NDCI.Metadata->getPropSets().find(NDCIMetadataKey);
  ASSERT_NE(NDCIMetadataIt, NDCI.Metadata->end());
  const PropertySet &NDCIMetadata = NDCIMetadataIt->second;
  ASSERT_EQ(NDCIMetadata.size(), size_t{1});

  // Make sure the arch string is preserved.
  SmallString<16> NDCIMetadataArchKey{"arch"};
  const auto &NDCIMetadataArchIt = NDCIMetadata.find(NDCIMetadataArchKey);
  ASSERT_NE(NDCIMetadataArchIt, NDCIMetadata.end());
  const PropertyValue &NDCIMetadataArch = NDCIMetadataArchIt->second;
  ASSERT_EQ(NDCIMetadataArch.getType(), PropertyValue::Type::BYTE_ARRAY);
  ASSERT_EQ(NDCIMetadataArch.getByteArraySize(), strlen(Arch));
  EXPECT_EQ(std::memcmp(NDCIMetadataArch.asByteArray(), Arch, strlen(Arch)), 0);

  // Check that the image is the same.
  ASSERT_EQ(Image.size(), NDCI.RawDeviceCodeImageBytes.size());
  EXPECT_EQ(std::memcmp(Image.data(), NDCI.RawDeviceCodeImageBytes.data(),
                        Image.size()),
            0);
}
