#include <sycl/sycl.hpp>

#include <detail/device_image_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/device_binary_image.hpp>
#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

class ProgramManagerExposed : public sycl::detail::ProgramManager
{
public:
   std::unordered_multimap<sycl::kernel_id, sycl::detail::RTDeviceBinaryImage *>& getKernelID2BinImage()
   {
    return m_KernelIDs2BinImage;
   }

   std::unordered_map<std::string, sycl::kernel_id>& getKernelName2KernelID()
   {
    return m_KernelName2KernelIDs;
   }

    std::unordered_map<sycl::detail::RTDeviceBinaryImage *,
      std::shared_ptr<std::vector<sycl::kernel_id>>>& getBinImage2KernelId()
      {
      return m_BinImg2KernelIDs;
      }

    std::unordered_multimap<std::string, sycl::detail::RTDeviceBinaryImage *>& getServiceKernels()
    {
      return m_ServiceKernels;
    }

    std::unordered_multimap<std::string, sycl::detail::RTDeviceBinaryImage *>& getExportedSymbolImages()
    {
      return m_ExportedSymbolImages;
    }
    
    std::unordered_map<sycl_device_binary, std::unique_ptr<sycl::detail::RTDeviceBinaryImage> >& getDeviceImages()
    {
      return m_DeviceImages;
    }

    std::unordered_map<std::string, std::set<sycl::detail::RTDeviceBinaryImage *> >& getVFSet2BinImage()
    {
      return m_VFSet2BinImage;
    }

std::unordered_multimap<ur_program_handle_t, const sycl::detail::RTDeviceBinaryImage *>& getNativePrograms()
{
  return NativePrograms;
}

 std::unordered_map<const sycl::detail::RTDeviceBinaryImage *, std::unordered_map<std::string, sycl::detail::KernelArgMask> >& getEliminatedKernelArgMask()
 {
  return m_EliminatedKernelArgMasks;
 }

 std::set<std::string>& getKernelUsesAssert()
 {
  return m_KernelUsesAssert;
 }

std::unordered_map<std::string, int>& getKernelImplicitLocalArgPos()
{
  return m_KernelImplicitLocalArgPos;
}

std::unordered_map<std::string, std::map<std::vector<unsigned char>, ur_kernel_handle_t>>& getMaterializedKernels()
{
  return m_MaterializedKernels;
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
  const std::string PropName = "virtual-functions-set";
  sycl::unittest::MockProperty Prop(PropName, Storage,
                                    SYCL_PROPERTY_TYPE_BYTE_ARRAY);

  Props.push_back(Prop);
  return Props;
}

std::string generateRefName(const std::string& ImageId, const std::string& FeatureName)
{
  return FeatureName + "_" + ImageId;
}

sycl::unittest::MockDeviceImage
generateImage(const std::string& ImageId) {
  sycl::unittest::MockPropertySet PropSet;

  std::initializer_list<std::string> KernelNames{ generateRefName(ImageId, "Kernel"), generateRefName(ImageId, "__sycl_service_kernel__") };
  const std::vector<std::string> ExportedSymbols{generateRefName(ImageId, "Exported")};
  const std::vector<std::string> &ImportedSymbols{generateRefName(ImageId, "Imported")};
  const std::string &VirtualFunctions{generateRefName(ImageId, "VF")};

    PropSet.insert(__SYCL_PROPERTY_SET_SYCL_EXPORTED_SYMBOLS,
                   createPropertySet(ExportedSymbols));

    PropSet.insert(__SYCL_PROPERTY_SET_SYCL_IMPORTED_SYMBOLS,
                   createPropertySet(ImportedSymbols));

    PropSet.insert(__SYCL_PROPERTY_SET_SYCL_VIRTUAL_FUNCTIONS,
                  createVFPropertySet(VirtualFunctions));
    setKernelUsesAssert(std::vector<std::string>{KernelNames.begin()[0]}, PropSet);

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
    generateImage("A"),
    generateImage("B")};
static std::array<sycl::unittest::MockDeviceImage, 1> ImagesToRemove = {
    generateImage("C")};

template <size_t ImageCount>
void convertAndAddImages(ProgramManagerExposed& PM, std::array<sycl::unittest::MockDeviceImage, ImageCount> Images, sycl_device_binary_struct* NativeImages, sycl_device_binaries_struct& AllBinaries)
{
    constexpr auto ImageSize = Images.size();
    for (size_t Idx = 0; Idx < ImageSize; ++Idx)
      NativeImages[Idx] = Images[Idx].convertToNativeType();

    AllBinaries = sycl_device_binaries_struct{
        SYCL_DEVICE_BINARIES_VERSION,
        ImageSize,
        NativeImages,
        nullptr,
        nullptr,
    };

    PM.addImages(&AllBinaries);
}

void checkAllInvolvedContainers(ProgramManagerExposed& PM, size_t ExpectedCount, const std::string& Comment)
{
    EXPECT_EQ(PM.getKernelID2BinImage().size(), ExpectedCount) << Comment;
    EXPECT_EQ(PM.getKernelName2KernelID().size(), ExpectedCount) << Comment;
    EXPECT_EQ(PM.getBinImage2KernelId().size(), ExpectedCount) << Comment;
  
    EXPECT_EQ(PM.getServiceKernels().size(), ExpectedCount) << Comment;

    EXPECT_EQ(PM.getExportedSymbolImages().size(), ExpectedCount) << Comment;
    EXPECT_EQ(PM.getDeviceImages().size(), ExpectedCount) << Comment;

   // EXPECT_EQ(PM.getVFSet2BinImage().size(), ExpectedCount) << Comment;
    EXPECT_EQ(PM.getNativePrograms().size(), 0u) << Comment;
    EXPECT_EQ(PM.getEliminatedKernelArgMask().size(), 0u) << Comment;

    EXPECT_EQ(PM.getKernelUsesAssert().size(), ExpectedCount) << Comment;
    EXPECT_EQ(PM.getKernelImplicitLocalArgPos().size(), 0u) << Comment;
}

TEST(ImageRemoval, Base) {
    ProgramManagerExposed PM;
    
    sycl_device_binary_struct NativeImages[ImagesToKeep.size()];
    sycl_device_binaries_struct AllBinaries;
    convertAndAddImages(PM, ImagesToKeep, NativeImages, AllBinaries);

    sycl_device_binary_struct NativeImagesForRemoval[ImagesToRemove.size()];
    sycl_device_binaries_struct TestBinaries;
    convertAndAddImages(PM, ImagesToRemove, NativeImagesForRemoval, TestBinaries);

    checkAllInvolvedContainers(PM, ImagesToRemove.size() + ImagesToKeep.size(), "Check failed before removal");

    PM.removeImages(&TestBinaries);

    checkAllInvolvedContainers(PM, ImagesToKeep.size(), "Check failed after removal");
}

} // anonymous namespace
