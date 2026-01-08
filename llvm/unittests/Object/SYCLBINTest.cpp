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

  // Create modules.
  SYCLBIN::SYCLBINDesc Desc{State, MDs};

  // Serialize the SYCLBIN.
  SmallString<0> SYCLBinary;
  raw_svector_ostream SYCLBinaryOS{SYCLBinary};
  if (Error E = SYCLBIN::write(Desc, SYCLBinaryOS))
    FAIL() << "Failed to write SYCLBIN.";

  // Deserialize the SYCLBIN.
  std::unique_ptr<MemoryBuffer> SYCBINDataBuffer =
      MemoryBuffer::getMemBuffer(SYCLBinary, /*BufferName=*/"",
                                 /*RequiresNullTerminator=*/false);
  Expected<std::unique_ptr<SYCLBIN>> SYCLBINObjOrError =
      SYCLBIN::read(*SYCBINDataBuffer);
  ASSERT_THAT_EXPECTED(SYCLBINObjOrError, Succeeded());
  SYCLBIN *SYCLBINObj = SYCLBINObjOrError->get();
  ASSERT_NE(SYCLBINObj, nullptr);

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

  // Currently we have an abstract module per image + 1 entry for global
  // metadata.
  ASSERT_EQ(SYCLBINObj->getOffloadBinaries().size(), size_t{NumImages + 1});
  ASSERT_EQ(SYCLBINObj->Metadata.size(), size_t{NumImages});

  std::vector<decltype(Images)::const_iterator> ImageIts;
  ImageIts.reserve(NumImages);
  size_t ObservedIRModules = 0, ObservedNativeDeviceCodeImages = 0;
  for (const std::unique_ptr<OffloadBinary> &OBPtr :
       SYCLBINObj->getOffloadBinaries()) {

    // Skip Global metadata entry.
    if (OBPtr->getFlags() & OIF_NoImage)
      continue;

    // This metadata should be the same as in the corresponding split module,
    // so testing should be expanded to ensure preservation.
    std::unique_ptr<llvm::util::PropertySetRegistry> &PSR =
        SYCLBINObj->Metadata[OBPtr.get()];
    ASSERT_NE(PSR.get(), nullptr);
    EXPECT_TRUE(PSR->getPropSets().empty());

    switch (OBPtr->getImageKind()) {
    case ImageKind::IMG_SPIRV:
      // The kind is currently locked to SPIR-V. This will change in the future.
      ++ObservedIRModules;
      break;
    case ImageKind::IMG_Object:
      ++ObservedNativeDeviceCodeImages;
      // Make sure the arch string is preserved.
      EXPECT_EQ(OBPtr->getString("arch"), StringRef(Arch));
      break;
    default:
      FAIL() << "Unexpected ImageKind: "
             << static_cast<int>(OBPtr->getImageKind());
      break;
    }

    // Make sure the triple string is preserved.
    EXPECT_EQ(OBPtr->getString("triple"), StringRef(IRMTarget));

    // Find the image that matches.
    std::vector<uint8_t> IRImage{OBPtr->getImage().begin(),
                                 OBPtr->getImage().end()};
    auto ImageMatchIt = std::find(Images.begin(), Images.end(), IRImage);
    ASSERT_NE(ImageMatchIt, Images.end());
    ImageIts.push_back(ImageMatchIt);
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
