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

    std::unordered_map<std::string, sycl::kernel_id>& getBuiltInKernelIds()
    {
      return m_BuiltInKernelIDs;
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

namespace CleanupImagesTest {
class KernelA1;
class KernelA2;
class KernelB;
} // namespace DynamicLinkingTest

namespace sycl {
inline namespace _V1 {
namespace detail {
#define KERNEL_INFO(KernelName)                                                \
  template <>                                                                  \
  struct KernelInfo<CleanupImagesTest::KernelName>                            \
      : public unittest::MockKernelInfoBase {                                  \
    static constexpr const char *getName() { return #KernelName; }             \
  };

KERNEL_INFO(KernelA1)
KERNEL_INFO(KernelA2)
KERNEL_INFO(KernelB)

#undef KERNEL_INFO

} // namespace detail
} // namespace _V1
} // namespace sycl

namespace {
std::vector<sycl::unittest::MockProperty>
createPropertySet(const std::vector<std::string> &Symbols) {
  sycl::unittest::MockPropertySet PropSet;
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

sycl::unittest::MockDeviceImage
generateImage(std::initializer_list<std::string> KernelNames,
              const std::vector<std::string> &ExportedSymbols,
              const std::vector<std::string> &ImportedSymbols,
              unsigned char Magic, sycl::detail::ur::DeviceBinaryType BinType,
              const char *DeviceTargetSpec,
              sycl::unittest::MockPropertySet PropSet) {
  if (!ExportedSymbols.empty())
    PropSet.insert(__SYCL_PROPERTY_SET_SYCL_EXPORTED_SYMBOLS,
                   createPropertySet(ExportedSymbols));
  if (!ImportedSymbols.empty())
    PropSet.insert(__SYCL_PROPERTY_SET_SYCL_IMPORTED_SYMBOLS,
                   createPropertySet(ImportedSymbols));

  std::vector<unsigned char> Bin{Magic};

  std::vector<sycl::unittest::MockOffloadEntry> Entries =
      sycl::unittest::makeEmptyKernels(KernelNames);

  sycl::unittest::MockDeviceImage Img{BinType,
                                      DeviceTargetSpec,
                                      "", // Compile options
                                      "", // Link options
                                      std::move(Bin),
                                      std::move(Entries),
                                      std::move(PropSet)};

  return Img;
}

constexpr size_t ImageCount = 2;
static sycl::unittest::MockDeviceImage Imgs[2] = {
    generateImage({"KernelA1", "KernelA2"}, {"ImageAExported"},
                  {"ImageAImported"}, 1, SYCL_DEVICE_BINARY_TYPE_NATIVE,
                  __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN, {}),
    generateImage({"KernelB"}, {"ImageBExported"},
                  {"ImageBImported"}, 2,
                  SYCL_DEVICE_BINARY_TYPE_NATIVE,
                  __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN, {})};

TEST(ImageRemoval, Base) {
    ProgramManagerExposed PM;
    
    sycl_device_binary_struct NativeImages[ImageCount];
    sycl_device_binaries_struct AllBinaries;

    for (size_t Idx = 0; Idx < ImageCount; ++Idx)
      NativeImages[Idx] = Imgs[Idx].convertToNativeType();

    AllBinaries = sycl_device_binaries_struct{
        SYCL_DEVICE_BINARIES_VERSION,
        ImageCount,
        NativeImages,
        nullptr, // not used, put here for compatibility with OpenMP
        nullptr, // not used, put here for compatibility with OpenMP
    };

    PM.addImages(&AllBinaries);
  
    EXPECT_EQ(PM.getKernelID2BinImage().size(), 3u);
    EXPECT_EQ(PM.getKernelName2KernelID().size(), 3u);
    EXPECT_EQ(PM.getBinImage2KernelId().size(), 2u);
  
    EXPECT_EQ(PM.getServiceKernels().size(), 0u);

    EXPECT_EQ(PM.getExportedSymbolImages().size(), 2u);
    EXPECT_EQ(PM.getDeviceImages().size(), ImageCount);
  
    EXPECT_EQ(PM.getBuiltInKernelIds().size(), 0u);
    EXPECT_EQ(PM.getVFSet2BinImage().size(), 0u);
    EXPECT_EQ(PM.getNativePrograms().size(), 0u);
    EXPECT_EQ(PM.getEliminatedKernelArgMask().size(), 0u);
    EXPECT_EQ(PM.getKernelUsesAssert().size(), 0u);
    EXPECT_EQ(PM.getKernelImplicitLocalArgPos().size(), 0u);
    EXPECT_EQ(PM.getMaterializedKernels().size(), 0u);

    sycl_device_binary_struct NativeImageToRemove = Imgs[0].convertToNativeType();
    sycl_device_binaries_struct BinaryToRemove = sycl_device_binaries_struct{
        SYCL_DEVICE_BINARIES_VERSION,
        1,
        &NativeImageToRemove,
        nullptr, // not used, put here for compatibility with OpenMP
        nullptr, // not used, put here for compatibility with OpenMP
    };

    PM.removeImages(&BinaryToRemove);

    EXPECT_EQ(PM.getKernelID2BinImage().size(), 1u);
    EXPECT_EQ(PM.getKernelName2KernelID().size(), 1u);
    EXPECT_EQ(PM.getBinImage2KernelId().size(), 1u);
  
    EXPECT_EQ(PM.getServiceKernels().size(), 0u);

    EXPECT_EQ(PM.getExportedSymbolImages().size(), 1u);
    EXPECT_EQ(PM.getDeviceImages().size(), ImageCount - 1);
  
    EXPECT_EQ(PM.getBuiltInKernelIds().size(), 0u);
    EXPECT_EQ(PM.getVFSet2BinImage().size(), 0u);
    EXPECT_EQ(PM.getNativePrograms().size(), 0u);
    EXPECT_EQ(PM.getEliminatedKernelArgMask().size(), 0u);
    EXPECT_EQ(PM.getKernelUsesAssert().size(), 0u);
    EXPECT_EQ(PM.getKernelImplicitLocalArgPos().size(), 0u);
    EXPECT_EQ(PM.getMaterializedKernels().size(), 0u);
}

} // anonymous namespace
