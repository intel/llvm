#include <sycl/sycl.hpp>

#include <detail/device_binary_image.hpp>
#include <detail/device_global_map.hpp>
#include <detail/device_image_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

class ProgramManagerExposed : public sycl::detail::ProgramManager {
public:
  std::unordered_multimap<sycl::kernel_id,
                          const sycl::detail::RTDeviceBinaryImage *> &
  getKernelID2BinImage() {
    return m_KernelIDs2BinImage;
  }

  std::unordered_map<sycl::detail::KernelNameStrT, sycl::kernel_id> &
  getKernelName2KernelID() {
    return m_KernelName2KernelIDs;
  }

  std::unordered_map<const sycl::detail::RTDeviceBinaryImage *,
                     std::shared_ptr<std::vector<sycl::kernel_id>>> &
  getBinImage2KernelId() {
    return m_BinImg2KernelIDs;
  }

  std::unordered_multimap<std::string,
                          const sycl::detail::RTDeviceBinaryImage *> &
  getExportedSymbolImages() {
    return m_ExportedSymbolImages;
  }

  std::unordered_map<sycl_device_binary,
                     std::unique_ptr<sycl::detail::RTDeviceBinaryImage>> &
  getDeviceImages() {
    return m_DeviceImages;
  }

  std::unordered_map<std::string,
                     std::set<const sycl::detail::RTDeviceBinaryImage *>> &
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

  std::unordered_map<sycl::detail::KernelNameStrT,
                     sycl::detail::DeviceKernelInfo> &
  getDeviceKernelInfoMap() {
    return m_DeviceKernelInfoMap;
  }

  std::unordered_map<sycl::detail::KernelNameStrT, int> &
  getKernelNameRefCount() {
    return m_KernelNameRefCount;
  }

  std::unordered_map<const sycl::detail::RTDeviceBinaryImage *,
                     std::unordered_map<sycl::detail::KernelNameStrT,
                                        sycl::detail::KernelArgMask>> &
  getEliminatedKernelArgMask() {
    return m_EliminatedKernelArgMasks;
  }

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

  sycl::detail::DeviceGlobalMap &getDeviceGlobals() { return m_DeviceGlobals; }
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

std::vector<std::string>
generateRefNames(const std::vector<std::string> &ImageIds,
                 const std::string &FeatureName) {
  std::vector<std::string> RefNames;
  RefNames.reserve(ImageIds.size());
  for (const std::string &ImageId : ImageIds)
    RefNames.push_back(generateRefName(ImageId, FeatureName));
  return RefNames;
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

sycl::unittest::MockDeviceImage generateImage(const std::string &ImageId,
                                              bool AddHostPipes = true) {
  sycl::unittest::MockPropertySet PropSet;

  std::initializer_list<std::string> KernelNames{
      generateRefName(ImageId, "Kernel")};
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

  PropSet.insert(__SYCL_PROPERTY_SET_SYCL_EXPORTED_SYMBOLS,
                 createPropertySet(ExportedSymbols));

  PropSet.insert(__SYCL_PROPERTY_SET_SYCL_IMPORTED_SYMBOLS,
                 createPropertySet(ImportedSymbols));

  PropSet.insert(__SYCL_PROPERTY_SET_SYCL_VIRTUAL_FUNCTIONS,
                 createVFPropertySet(VirtualFunctions));

  PropSet.insert(__SYCL_PROPERTY_SET_SYCL_IMPLICIT_LOCAL_ARG,
                 createPropertySet(ImplicitLocalArg));
  PropSet.insert(__SYCL_PROPERTY_SET_KERNEL_PARAM_OPT_INFO, std::move(ImgKPOI));

  PropSet.insert(
      __SYCL_PROPERTY_SET_SYCL_DEVICE_GLOBALS,
      std::vector<sycl::unittest::MockProperty>{
          sycl::unittest::makeDeviceGlobalInfo(
              generateRefName(ImageId, "DeviceGlobal"), sizeof(int), 0)});
  if (AddHostPipes)
    PropSet.insert(__SYCL_PROPERTY_SET_SYCL_HOST_PIPES,
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

static std::array<sycl::unittest::MockDeviceImage, 1> ImagesToKeepSameEntries =
    {generateImage("A", /*AddHostPipe*/ false)};
static std::array<sycl::unittest::MockDeviceImage, 1>
    ImagesToRemoveSameEntries = {generateImage("A", /*AddHostPipe*/ false)};

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

template <typename T>
void checkContainer(const T &Container, size_t ExpectedCount,
                    const std::vector<std::string> &ExpectedEntries,
                    const std::string &Comment) {
  EXPECT_EQ(Container.size(), ExpectedCount) << Comment;
  for (const std::string &Entry : ExpectedEntries) {
    EXPECT_TRUE(Container.count(Entry) > 0) << Comment;
  }
}

void checkAllInvolvedContainers(ProgramManagerExposed &PM,
                                size_t ExpectedImgCount,
                                size_t ExpectedEntryCount,
                                const std::vector<std::string> &ImgIds,
                                const std::string &CommentPostfix,
                                bool MultipleImgsPerEntryTestCase = false) {
  EXPECT_EQ(PM.getKernelID2BinImage().size(), ExpectedImgCount)
      << "KernelID2BinImg " + CommentPostfix;
  checkContainer(PM.getKernelName2KernelID(), ExpectedEntryCount,
                 generateRefNames(ImgIds, "Kernel"),
                 "KernelName2KernelID " + CommentPostfix);
  EXPECT_EQ(PM.getBinImage2KernelId().size(), ExpectedImgCount)
      << CommentPostfix;
  checkContainer(PM.getExportedSymbolImages(), ExpectedImgCount,
                 generateRefNames(ImgIds, "Exported"),
                 "Exported symbol images " + CommentPostfix);
  EXPECT_EQ(PM.getDeviceImages().size(), ExpectedImgCount)
      << "Device images " + CommentPostfix;

  checkContainer(PM.getVFSet2BinImage(), ExpectedEntryCount,
                 generateRefNames(ImgIds, "VF"),
                 "VFSet2BinImage " + CommentPostfix);
  checkContainer(PM.getDeviceKernelInfoMap(), ExpectedEntryCount,
                 generateRefNames(ImgIds, "Kernel"),
                 "Device kernel info map " + CommentPostfix);
  checkContainer(PM.getKernelNameRefCount(), ExpectedEntryCount,
                 generateRefNames(ImgIds, "Kernel"),
                 "Kernel name reference count " + CommentPostfix);
  EXPECT_EQ(PM.getEliminatedKernelArgMask().size(), ExpectedImgCount)
      << "Eliminated kernel arg mask " + CommentPostfix;
  EXPECT_EQ(PM.getKernelImplicitLocalArgPos().size(), ExpectedEntryCount)
      << "Kernel implicit local arg pos " + CommentPostfix;

  if (!MultipleImgsPerEntryTestCase) {
    // FIXME expected to fail for now, device globals cleanup seems to be
    // purging all info for symbols associated with the removed image.
    checkContainer(PM.getDeviceGlobals(), ExpectedEntryCount,
                   generateRefNames(ImgIds, "DeviceGlobal"),
                   "Device globals " + CommentPostfix);

    // The test case with the same entries in multiple images doesn't support
    // host pipes since those are assumed to be unique.
    checkContainer(PM.getHostPipes(), ExpectedEntryCount,
                   generateRefNames(ImgIds, "HostPipe"),
                   "Host pipes " + CommentPostfix);
    EXPECT_EQ(PM.getPtrToHostPipe().size(), ExpectedEntryCount)
        << "Pointer to host pipe " + CommentPostfix;
  }
}

void checkAllInvolvedContainers(ProgramManagerExposed &PM, size_t ExpectedCount,
                                const std::vector<std::string> &ImgIds,
                                const std::string &CommentPostfix,
                                bool CheckHostPipes = false) {
  checkAllInvolvedContainers(PM, ExpectedCount, ExpectedCount, ImgIds,
                             CommentPostfix, CheckHostPipes);
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
  std::vector<std::string> KernelNames =
      generateRefNames({"A", "B", "C"}, "Kernel");
  for (const std::string &Name : KernelNames)
    PM.getOrCreateDeviceKernelInfo(Name);

  checkAllInvolvedContainers(PM, ImagesToRemove.size() + ImagesToKeep.size(),
                             {"A", "B", "C"}, "check failed before removal");

  PM.removeImages(&TestBinaries);

  checkAllInvolvedContainers(PM, ImagesToKeep.size(), {"A", "B"},
                             "check failed after removal");
}

TEST(ImageRemoval, MultipleImagesPerEntry) {
  ProgramManagerExposed PM;

  sycl_device_binary_struct NativeImages[ImagesToKeepSameEntries.size()];
  sycl_device_binaries_struct AllBinaries;
  convertAndAddImages(PM, ImagesToKeepSameEntries, NativeImages, AllBinaries);

  sycl_device_binary_struct
      NativeImagesForRemoval[ImagesToRemoveSameEntries.size()];
  sycl_device_binaries_struct TestBinaries;
  convertAndAddImages(PM, ImagesToRemoveSameEntries, NativeImagesForRemoval,
                      TestBinaries);

  std::string KernelName = generateRefName("A", "Kernel");
  PM.getOrCreateDeviceKernelInfo(KernelName);
  checkAllInvolvedContainers(
      PM, ImagesToRemoveSameEntries.size() + ImagesToKeepSameEntries.size(),
      /*ExpectedEntryCount*/ 1, {"A"}, "check failed before removal",
      /*MultipleImgsPerEntryTestCase*/ true);

  PM.removeImages(&TestBinaries);

  checkAllInvolvedContainers(PM, ImagesToKeepSameEntries.size(), {"A"},
                             "check failed after removal",
                             /*MultipleImgsPerEntryTestCase*/ true);
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
  auto ProgramA = PM.getBuiltURProgram(*sycl::detail::getSyclObjImpl(Ctx),
                                       *sycl::detail::getSyclObjImpl(Dev),
                                       generateRefName("A", "Kernel"));
  auto ProgramB = PM.getBuiltURProgram(*sycl::detail::getSyclObjImpl(Ctx),
                                       *sycl::detail::getSyclObjImpl(Dev),
                                       generateRefName("B", "Kernel"));
  std::ignore = PM.getBuiltURProgram(*sycl::detail::getSyclObjImpl(Ctx),
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
