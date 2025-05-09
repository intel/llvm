#include <sycl/sycl.hpp>

#include <detail/device_binary_image.hpp>
#include <detail/device_image_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

class ProgramManagerExposed : public sycl::detail::ProgramManager {
public:
  std::unordered_multimap<sycl::kernel_id,
                          sycl::detail::RTDeviceBinaryImage *> &
  getKernelID2BinImage() {
    return m_KernelIDs2BinImage;
  }

  std::unordered_map<sycl::detail::KernelNameStrT, sycl::kernel_id> &
  getKernelName2KernelID() {
    return m_KernelName2KernelIDs;
  }

  std::unordered_map<sycl::detail::RTDeviceBinaryImage *,
                     std::shared_ptr<std::vector<sycl::kernel_id>>> &
  getBinImage2KernelId() {
    return m_BinImg2KernelIDs;
  }

  std::unordered_multimap<sycl::detail::KernelNameStrT,
                          sycl::detail::RTDeviceBinaryImage *> &
  getServiceKernels() {
    return m_ServiceKernels;
  }

  std::unordered_multimap<sycl::detail::KernelNameStrT,
                          sycl::detail::RTDeviceBinaryImage *> &
  getExportedSymbolImages() {
    return m_ExportedSymbolImages;
  }

  std::unordered_map<sycl_device_binary,
                     std::unique_ptr<sycl::detail::RTDeviceBinaryImage>> &
  getDeviceImages() {
    return m_DeviceImages;
  }

  std::unordered_map<std::string,
                     std::set<sycl::detail::RTDeviceBinaryImage *>> &
  getVFSet2BinImage() {
    return m_VFSet2BinImage;
  }

  std::unordered_multimap<
      ur_program_handle_t,
      std::pair<std::weak_ptr<sycl::detail::context_impl>,
                const sycl::detail::RTDeviceBinaryImage *>> &
  getNativePrograms() {
    return NativePrograms;
  }

  std::unordered_map<const sycl::detail::RTDeviceBinaryImage *,
                     std::unordered_map<sycl::detail::KernelNameStrT,
                                        sycl::detail::KernelArgMask>> &
  getEliminatedKernelArgMask() {
    return m_EliminatedKernelArgMasks;
  }

  KernelUsesAssertSet &getKernelUsesAssert() { return m_KernelUsesAssert; }

  std::unordered_map<sycl::detail::KernelNameStrT, int> &
  getKernelImplicitLocalArgPos() {
    return m_KernelImplicitLocalArgPos;
  }

  std::unordered_map<std::string,
                     std::unique_ptr<sycl::detail::HostPipeMapEntry>> &
  getHostPipes() {
    return m_HostPipes;
  }

  std::unordered_map<const void *, sycl::detail::HostPipeMapEntry *> &
  getPtrToHostPipe() {
    return m_Ptr2HostPipe;
  }

  std::unordered_map<sycl::detail::KernelNameStrT,
                     std::unique_ptr<sycl::detail::DeviceGlobalMapEntry>> &
  getDeviceGlobals() {
    return m_DeviceGlobals;
  }

  std::unordered_map<const void *, sycl::detail::DeviceGlobalMapEntry *> &
  getPtrToDeviceGlobal() {
    return m_Ptr2DeviceGlobal;
  }
};

namespace {
std::vector<sycl::unittest::MockProperty>
createPropertySet(const std::vector<std::string> &Symbols) {
  std::vector<sycl::unittest::MockProperty> Props;
  for (const std::string &Symbol : Symbols) {
    std::vector<char> Storage(sizeof(uint32_t));
    uint32_t Val = 1;
    auto *DataPtr = reinterpret_cast<char *>(&Val);
    std::uninitialized_copy(DataPtr, DataPtr + sizeof(uint32_t),
                            Storage.data());

    sycl::unittest::MockProperty Prop(Symbol, Storage,
                                      SYCL_PROPERTY_TYPE_UINT32);

    Props.push_back(Prop);
  }
  return Props;
}

std::vector<sycl::unittest::MockProperty>
createVFPropertySet(const std::string &VFSets) {
  std::vector<sycl::unittest::MockProperty> Props;
  uint64_t PropSize = VFSets.size();
  std::vector<char> Storage(/* bytes for size */ 8 + PropSize +
                            /* null terminator */ 1);
  auto *SizePtr = reinterpret_cast<char *>(&PropSize);
  std::uninitialized_copy(SizePtr, SizePtr + sizeof(uint64_t), Storage.data());
  std::uninitialized_copy(VFSets.data(), VFSets.data() + PropSize,
                          Storage.data() + /* bytes for size */ 8);
  Storage.back() = '\0';
  const std::string PropName = "uses-virtual-functions-set";
  sycl::unittest::MockProperty Prop(PropName, Storage,
                                    SYCL_PROPERTY_TYPE_BYTE_ARRAY);

  Props.push_back(Prop);
  return Props;
}

std::string generateRefName(const std::string &ImageId,
                            const std::string &FeatureName) {
  return FeatureName + "_" + ImageId;
}

sycl::ext::oneapi::experimental::device_global<int> DeviceGlobalA;
sycl::ext::oneapi::experimental::device_global<int> DeviceGlobalB;
sycl::ext::oneapi::experimental::device_global<int> DeviceGlobalC;

class PipeIDA;
class PipeIDB;
class PipeIDC;
using PipeA = sycl::ext::intel::experimental::pipe<PipeIDA, int, 10>;
using PipeB = sycl::ext::intel::experimental::pipe<PipeIDB, int, 10>;
using PipeC = sycl::ext::intel::experimental::pipe<PipeIDC, int, 10>;

sycl::unittest::MockDeviceImage generateImage(const std::string &ImageId) {
  sycl::unittest::MockPropertySet PropSet;

  std::initializer_list<std::string> KernelNames{
      generateRefName(ImageId, "Kernel"),
      generateRefName(ImageId, "__sycl_service_kernel__")};
  const std::vector<std::string> ExportedSymbols{
      generateRefName(ImageId, "Exported")};
  const std::vector<std::string> ImportedSymbols{
      generateRefName(ImageId, "Imported")};
  const std::vector<std::string> ImplicitLocalArg{KernelNames.begin()[0]};
  const std::string &VirtualFunctions{generateRefName(ImageId, "VF")};
  std::vector<unsigned char> KernelEAM{0b0000001};
  sycl::unittest::MockProperty EAMKernelPOI =
      sycl::unittest::makeKernelParamOptInfo(KernelNames.begin()[0], 1,
                                             KernelEAM);
  std::vector<sycl::unittest::MockProperty> ImgKPOI{std::move(EAMKernelPOI)};

  PropSet.insert(llvm::util::PropertySetRegistry::SYCL_EXPORTED_SYMBOLS,
                 createPropertySet(ExportedSymbols));

  PropSet.insert(llvm::util::PropertySetRegistry::SYCL_IMPORTED_SYMBOLS,
                 createPropertySet(ImportedSymbols));

  PropSet.insert(llvm::util::PropertySetRegistry::SYCL_VIRTUAL_FUNCTIONS,
                 createVFPropertySet(VirtualFunctions));
  setKernelUsesAssert(std::vector<std::string>{KernelNames.begin()[0]},
                      PropSet);

  PropSet.insert(llvm::util::PropertySetRegistry::SYCL_IMPLICIT_LOCAL_ARG,
                 createPropertySet(ImplicitLocalArg));
  PropSet.insert(llvm::util::PropertySetRegistry::SYCL_KERNEL_PARAM_OPT_INFO,
                 std::move(ImgKPOI));

  PropSet.insert(
      llvm::util::PropertySetRegistry::SYCL_DEVICE_GLOBALS,
      std::vector<sycl::unittest::MockProperty>{
          sycl::unittest::makeDeviceGlobalInfo(
              generateRefName(ImageId, "DeviceGlobal"), sizeof(int), 0)});

  PropSet.insert(llvm::util::PropertySetRegistry::SYCL_HOST_PIPES,
                 std::vector<sycl::unittest::MockProperty>{
                     sycl::unittest::makeHostPipeInfo(
                         generateRefName(ImageId, "HostPipe"), sizeof(int))});
  std::vector<unsigned char> Bin{0};

  std::vector<sycl::unittest::MockOffloadEntry> Entries =
      sycl::unittest::makeEmptyKernels(KernelNames);

  sycl::unittest::MockDeviceImage Img{SYCL_DEVICE_BINARY_TYPE_NATIVE,
                                      __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN,
                                      "", // Compile options
                                      "", // Link options
                                      std::move(Bin),
                                      std::move(Entries),
                                      std::move(PropSet)};

  return Img;
}

sycl::unittest::MockDeviceImage
generateImageKernelOnly(const std::string &ImageId) {
  sycl::unittest::MockPropertySet PropSet;

  std::initializer_list<std::string> KernelNames{
      generateRefName(ImageId, "Kernel")};
  std::vector<unsigned char> Bin{0};

  std::vector<sycl::unittest::MockOffloadEntry> Entries =
      sycl::unittest::makeEmptyKernels(KernelNames);

  sycl::unittest::MockDeviceImage Img{SYCL_DEVICE_BINARY_TYPE_NATIVE,
                                      __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN,
                                      "", // Compile options
                                      "", // Link options
                                      std::move(Bin),
                                      std::move(Entries),
                                      std::move(PropSet)};

  return Img;
}

static std::array<sycl::unittest::MockDeviceImage, 2> ImagesToKeep = {
    generateImage("A"), generateImage("B")};
static std::array<sycl::unittest::MockDeviceImage, 1> ImagesToRemove = {
    generateImage("C")};

static std::array<sycl::unittest::MockDeviceImage, 2> ImagesToKeepKernelOnly = {
    generateImageKernelOnly("A"), generateImageKernelOnly("B")};
static std::array<sycl::unittest::MockDeviceImage, 1> ImagesToRemoveKernelOnly =
    {generateImageKernelOnly("C")};

template <size_t ImageCount>
void convertAndAddImages(
    ProgramManagerExposed &PM,
    std::array<sycl::unittest::MockDeviceImage, ImageCount> Images,
    sycl_device_binary_struct *NativeImages,
    sycl_device_binaries_struct &AllBinaries) {
  constexpr auto ImageSize = Images.size();
  for (size_t Idx = 0; Idx < ImageSize; ++Idx)
    NativeImages[Idx] = Images[Idx].convertToNativeType();

  AllBinaries = sycl_device_binaries_struct{
      SYCL_DEVICE_BINARIES_VERSION, ImageSize, NativeImages, nullptr, nullptr,
  };

  PM.addImages(&AllBinaries);
}

void checkAllInvolvedContainers(ProgramManagerExposed &PM, size_t ExpectedCount,
                                const std::string &Comment) {
  EXPECT_EQ(PM.getKernelID2BinImage().size(), ExpectedCount) << Comment;
  {
    EXPECT_EQ(PM.getKernelName2KernelID().size(), ExpectedCount) << Comment;
    EXPECT_TRUE(
        PM.getKernelName2KernelID().count(generateRefName("A", "Kernel")) > 0)
        << Comment;
    EXPECT_TRUE(
        PM.getKernelName2KernelID().count(generateRefName("B", "Kernel")) > 0)
        << Comment;
  }
  EXPECT_EQ(PM.getBinImage2KernelId().size(), ExpectedCount) << Comment;
  {
    EXPECT_EQ(PM.getServiceKernels().size(), ExpectedCount) << Comment;
    EXPECT_TRUE(PM.getServiceKernels().count(
                    generateRefName("A", "__sycl_service_kernel__")) > 0)
        << Comment;
    EXPECT_TRUE(PM.getServiceKernels().count(
                    generateRefName("B", "__sycl_service_kernel__")) > 0)
        << Comment;
  }
  {
    EXPECT_EQ(PM.getExportedSymbolImages().size(), ExpectedCount) << Comment;
    EXPECT_TRUE(PM.getExportedSymbolImages().count(
                    generateRefName("A", "Exported")) > 0)
        << Comment;
    EXPECT_TRUE(PM.getExportedSymbolImages().count(
                    generateRefName("B", "Exported")) > 0)
        << Comment;
  }
  EXPECT_EQ(PM.getDeviceImages().size(), ExpectedCount) << Comment;
  {
    EXPECT_EQ(PM.getVFSet2BinImage().size(), ExpectedCount) << Comment;
    EXPECT_TRUE(PM.getVFSet2BinImage().count(generateRefName("A", "VF")) > 0)
        << Comment;
    EXPECT_TRUE(PM.getVFSet2BinImage().count(generateRefName("B", "VF")) > 0)
        << Comment;
  }

  EXPECT_EQ(PM.getEliminatedKernelArgMask().size(), ExpectedCount) << Comment;
  {
    EXPECT_EQ(PM.getKernelUsesAssert().size(), ExpectedCount) << Comment;
    EXPECT_TRUE(PM.getKernelUsesAssert().count(generateRefName("A", "Kernel")) >
                0)
        << Comment;
    EXPECT_TRUE(PM.getKernelUsesAssert().count(generateRefName("B", "Kernel")) >
                0)
        << Comment;
  }
  EXPECT_EQ(PM.getKernelImplicitLocalArgPos().size(), ExpectedCount) << Comment;

  {
    EXPECT_EQ(PM.getDeviceGlobals().size(), ExpectedCount) << Comment;
    EXPECT_TRUE(
        PM.getDeviceGlobals().count(generateRefName("A", "DeviceGlobal")) > 0)
        << Comment;
    EXPECT_TRUE(
        PM.getDeviceGlobals().count(generateRefName("B", "DeviceGlobal")) > 0)
        << Comment;
  }
  EXPECT_EQ(PM.getPtrToDeviceGlobal().size(), ExpectedCount) << Comment;

  {
    EXPECT_EQ(PM.getHostPipes().size(), ExpectedCount) << Comment;
    EXPECT_TRUE(PM.getHostPipes().count(generateRefName("A", "HostPipe")) > 0)
        << Comment;
    EXPECT_TRUE(PM.getHostPipes().count(generateRefName("B", "HostPipe")) > 0)
        << Comment;
  }
  EXPECT_EQ(PM.getPtrToHostPipe().size(), ExpectedCount) << Comment;
}

TEST(ImageRemoval, BaseContainers) {
  ProgramManagerExposed PM;

  sycl_device_binary_struct NativeImages[ImagesToKeep.size()];
  sycl_device_binaries_struct AllBinaries;
  convertAndAddImages(PM, ImagesToKeep, NativeImages, AllBinaries);

  sycl_device_binary_struct NativeImagesForRemoval[ImagesToRemove.size()];
  sycl_device_binaries_struct TestBinaries;
  convertAndAddImages(PM, ImagesToRemove, NativeImagesForRemoval, TestBinaries);

  PM.addOrInitDeviceGlobalEntry(&DeviceGlobalA,
                                generateRefName("A", "DeviceGlobal").c_str());
  PM.addOrInitDeviceGlobalEntry(&DeviceGlobalB,
                                generateRefName("B", "DeviceGlobal").c_str());
  PM.addOrInitDeviceGlobalEntry(&DeviceGlobalC,
                                generateRefName("C", "DeviceGlobal").c_str());
  PM.addOrInitHostPipeEntry(PipeA::get_host_ptr(),
                            generateRefName("A", "HostPipe").c_str());
  PM.addOrInitHostPipeEntry(PipeB::get_host_ptr(),
                            generateRefName("B", "HostPipe").c_str());
  PM.addOrInitHostPipeEntry(PipeC::get_host_ptr(),
                            generateRefName("C", "HostPipe").c_str());

  checkAllInvolvedContainers(PM, ImagesToRemove.size() + ImagesToKeep.size(),
                             "Check failed before removal");

  PM.removeImages(&TestBinaries);

  checkAllInvolvedContainers(PM, ImagesToKeep.size(),
                             "Check failed after removal");
}

TEST(ImageRemoval, NativePrograms) {
  ProgramManagerExposed PM;

  sycl_device_binary_struct NativeImages[ImagesToKeepKernelOnly.size()];
  sycl_device_binaries_struct AllBinaries;
  convertAndAddImages(PM, ImagesToKeepKernelOnly, NativeImages, AllBinaries);

  sycl_device_binary_struct
      NativeImagesForRemoval[ImagesToRemoveKernelOnly.size()];
  sycl_device_binaries_struct TestBinaries;
  convertAndAddImages(PM, ImagesToRemoveKernelOnly, NativeImagesForRemoval,
                      TestBinaries);

  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  const sycl::device Dev = Plt.get_devices()[0];
  sycl::queue Queue{Dev};
  auto Ctx = Queue.get_context();
  auto ProgramA = PM.getBuiltURProgram(sycl::detail::getSyclObjImpl(Ctx),
                                       *sycl::detail::getSyclObjImpl(Dev),
                                       generateRefName("A", "Kernel"));
  auto ProgramB = PM.getBuiltURProgram(sycl::detail::getSyclObjImpl(Ctx),
                                       *sycl::detail::getSyclObjImpl(Dev),
                                       generateRefName("B", "Kernel"));
  std::ignore = PM.getBuiltURProgram(sycl::detail::getSyclObjImpl(Ctx),
                                     *sycl::detail::getSyclObjImpl(Dev),
                                     generateRefName("C", "Kernel"));

  EXPECT_EQ(PM.getNativePrograms().size(),
            ImagesToRemoveKernelOnly.size() + ImagesToKeepKernelOnly.size());

  PM.removeImages(&TestBinaries);

  EXPECT_EQ(PM.getNativePrograms().size(), ImagesToKeepKernelOnly.size());
  EXPECT_TRUE(PM.getNativePrograms().count(ProgramA) > 0);
  EXPECT_TRUE(PM.getNativePrograms().count(ProgramB) > 0);
}
} // anonymous namespace
