#include <sycl/sycl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/RuntimeLinkingCommon.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

// Check that the dependencies of an AOT image are resolved with images built
// for the same device when there are images for several devices.

namespace DynamicLinkingCompileTargetTest {
class MultiArchKernel;
} // namespace DynamicLinkingCompileTargetTest

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<DynamicLinkingCompileTargetTest::MultiArchKernel>
    : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "MultiArchKernel"; }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

namespace {
static constexpr unsigned TGLLP_PRG = 31;
static constexpr unsigned TGLLP_PRG_DEP = 37;
static constexpr unsigned PVC_PRG = 41;
static constexpr unsigned PVC_PRG_DEP = 43;

// IP version of a PVC device.
static constexpr uint32_t PvcIp = 0x030f0000;

sycl::unittest::MockDeviceImage
generateImage(std::initializer_list<std::string> KernelNames,
              const std::string &ExportedSymbol,
              const std::string &ImportedSymbol, unsigned char Magic,
              const std::string &CompileTarget) {
  sycl::unittest::MockPropertySet PropSet;

  std::vector<char> Data(8 + CompileTarget.size());
  std::copy(CompileTarget.begin(), CompileTarget.end(), Data.data() + 8);
  PropSet.insert(__SYCL_PROPERTY_SET_SYCL_DEVICE_REQUIREMENTS,
                 sycl::unittest::MockProperty("compile_target", Data,
                                              SYCL_PROPERTY_TYPE_BYTE_ARRAY));

  auto makeSymbolProps = [](const std::string &Symbol) {
    std::vector<sycl::unittest::MockProperty> Props;
    Props.emplace_back(Symbol, std::vector<char>(sizeof(uint32_t)),
                       SYCL_PROPERTY_TYPE_UINT32);
    return Props;
  };
  if (!ExportedSymbol.empty())
    PropSet.insert(__SYCL_PROPERTY_SET_SYCL_EXPORTED_SYMBOLS,
                   makeSymbolProps(ExportedSymbol));
  if (!ImportedSymbol.empty())
    PropSet.insert(__SYCL_PROPERTY_SET_SYCL_IMPORTED_SYMBOLS,
                   makeSymbolProps(ImportedSymbol));

  return sycl::unittest::MockDeviceImage{
      SYCL_DEVICE_BINARY_TYPE_NATIVE,
      __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN,
      "", // Compile options
      "", // Link options
      std::vector<unsigned char>{Magic},
      sycl::unittest::makeEmptyKernels(KernelNames),
      std::move(PropSet)};
}

// The TGLLP images come first, so that picking the first image exporting the
// symbol would choose the wrong dependency for a PVC device.
static sycl::unittest::MockDeviceImage Imgs[] = {
    generateImage({"MultiArchKernel"}, "", "MultiArchKernelDep", TGLLP_PRG,
                  "intel_gpu_tgllp"),
    generateImage({"MultiArchKernelDep"}, "MultiArchKernelDep", "",
                  TGLLP_PRG_DEP, "intel_gpu_tgllp"),
    generateImage({"MultiArchKernel"}, "", "MultiArchKernelDep", PVC_PRG,
                  "intel_gpu_pvc"),
    generateImage({"MultiArchKernelDep"}, "MultiArchKernelDep", "", PVC_PRG_DEP,
                  "intel_gpu_pvc")};
static sycl::unittest::MockDeviceImageArray<4> ImgArray{Imgs};

ur_result_t redefinedDeviceGetInfo(void *pParams) {
  auto Params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*Params.ppropName == UR_DEVICE_INFO_IP_VERSION && *Params.ppPropValue)
    *static_cast<uint32_t *>(*Params.ppPropValue) = PvcIp;
  if (*Params.ppropName == UR_DEVICE_INFO_TYPE && *Params.ppPropValue)
    *static_cast<ur_device_type_t *>(*Params.ppPropValue) = UR_DEVICE_TYPE_GPU;
  return UR_RESULT_SUCCESS;
}
} // namespace

TEST(DynamicLinking, AheadOfTimeDependencyForDevice) {
  sycl::unittest::UrMock<sycl::backend::ext_oneapi_level_zero> Mock;
  setupRuntimeLinkingMock();
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfo);

  sycl::platform Plt = sycl::platform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedLinkingData.clear();

  Q.single_task<DynamicLinkingCompileTargetTest::MultiArchKernel>([=]() {});
  ASSERT_EQ(CapturedLinkingData.NumOfUrProgramLinkCalls, 1u);
  EXPECT_TRUE(
      CapturedLinkingData.LinkedProgramsContains({PVC_PRG, PVC_PRG_DEP}));
  EXPECT_EQ(CapturedLinkingData.ProgramUsedToCreateKernel,
            PVC_PRG * PVC_PRG_DEP);
}

TEST(DynamicLinking, AheadOfTimeKernelBundleForDevice) {
  sycl::unittest::UrMock<sycl::backend::ext_oneapi_level_zero> Mock;
  setupRuntimeLinkingMock();
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfo);

  sycl::platform Plt = sycl::platform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedLinkingData.clear();

  auto Bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      Q.get_context(), {Q.get_device()},
      {sycl::get_kernel_id<
          DynamicLinkingCompileTargetTest::MultiArchKernel>()});
  Q.submit([&](sycl::handler &CGH) {
    CGH.use_kernel_bundle(Bundle);
    CGH.single_task<DynamicLinkingCompileTargetTest::MultiArchKernel>([=]() {});
  });
  ASSERT_EQ(CapturedLinkingData.NumOfUrProgramLinkCalls, 1u);
  EXPECT_TRUE(
      CapturedLinkingData.LinkedProgramsContains({PVC_PRG, PVC_PRG_DEP}));
  EXPECT_EQ(CapturedLinkingData.ProgramUsedToCreateKernel,
            PVC_PRG * PVC_PRG_DEP);
}
