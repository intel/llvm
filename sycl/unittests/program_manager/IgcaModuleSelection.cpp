// Checks that a kernel defined in a module built in --offload-arch mode is
// taken from that module without ranking, that modules not built that way keep
// using the existing selection, and that a module the backend refuses names the
// device and the module's IGCA targets.

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <algorithm>

using namespace sycl;

namespace {

std::vector<char> byteArrayValue(const std::string &Value) {
  // A byte array property value is preceded by an 8 byte size header.
  std::vector<char> Data(8 + Value.size());
  std::copy(Value.begin(), Value.end(), Data.data() + 8);
  return Data;
}

std::vector<char> uint32Value(uint32_t Value) {
  std::vector<char> Data(sizeof(uint32_t));
  std::memcpy(Data.data(), &Value, sizeof(Value));
  return Data;
}

// An AOT module built in --offload-arch mode: native code concatenated from a
// sub-module per IGCA target. Its binary is the kernel name so that the test
// can tell which image reached the backend.
unittest::MockDeviceImage generateIgcaModule(
    const std::string &KernelName, const std::string &IgcaTargets,
    bool RecordTargets = true,
    const char *DeviceTargetSpec = __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN,
    const std::string &CompileTarget = "") {
  unittest::MockPropertySet PropSet;
  std::vector<unittest::MockProperty> Props;
  Props.emplace_back(__SYCL_IGCA_METADATA_IS_IGCA_MODULE, uint32Value(1),
                     SYCL_PROPERTY_TYPE_UINT32);
  if (RecordTargets)
    Props.emplace_back(__SYCL_IGCA_METADATA_TARGETS,
                       byteArrayValue(IgcaTargets),
                       SYCL_PROPERTY_TYPE_BYTE_ARRAY);
  PropSet.insert(__SYCL_PROPERTY_SET_SYCL_IGCA_METADATA, std::move(Props));
  if (!CompileTarget.empty())
    PropSet.insert(__SYCL_PROPERTY_SET_SYCL_DEVICE_REQUIREMENTS,
                   unittest::MockProperty{"compile_target",
                                          byteArrayValue(CompileTarget),
                                          SYCL_PROPERTY_TYPE_BYTE_ARRAY});

  std::vector<unsigned char> Bin(KernelName.begin(), KernelName.end());
  Bin.push_back(0);

  return unittest::MockDeviceImage{SYCL_DEVICE_BINARY_TYPE_NATIVE,
                                   DeviceTargetSpec,
                                   "",
                                   "",
                                   std::move(Bin),
                                   unittest::makeEmptyKernels({KernelName}),
                                   std::move(PropSet)};
}

} // namespace

class IgcaOnlyKernel;
class IgcaAndJitKernel;
class JitOnlyKernel;
class IgcaOtherBackendKernel;
class IgcaBuildFailureKernel;
class IgcaNoRecordedTargetsKernel;

MOCK_INTEGRATION_HEADER(IgcaOnlyKernel)
MOCK_INTEGRATION_HEADER(IgcaAndJitKernel)
MOCK_INTEGRATION_HEADER(JitOnlyKernel)
MOCK_INTEGRATION_HEADER(IgcaOtherBackendKernel)
MOCK_INTEGRATION_HEADER(IgcaBuildFailureKernel)
MOCK_INTEGRATION_HEADER(IgcaNoRecordedTargetsKernel)

static unittest::MockDeviceImage Imgs[] = {
    // An uber-module whose sub-modules were AOT compiled, so it carries a
    // compile_target that describes only one of them and does not match the
    // mock device.
    generateIgcaModule("IgcaOnlyKernel", "igca_10,igca_20",
                       /*RecordTargets=*/true,
                       __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN,
                       /*CompileTarget=*/"intel_gpu_pvc"),
    // The same kernel in a module built in --offload-arch mode and in a JIT
    // module, to check which of the two is picked.
    generateIgcaModule("IgcaAndJitKernel", "igca_10"),
    unittest::generateDefaultImage({"IgcaAndJitKernel"}),
    // A kernel with no --offload-arch module involved, in two images the
    // backend has to rank: selection must be unchanged for it, even though the
    // binary holds --offload-arch modules for other kernels.
    unittest::generateDefaultImage({"JitOnlyKernel"}),
    unittest::generateDefaultImage({"JitOnlyKernel"}),
    generateIgcaModule("IgcaBuildFailureKernel", "igca_20"),
    generateIgcaModule("IgcaNoRecordedTargetsKernel", "igca_20",
                       /*RecordTargets=*/false),
    // A module that no backend of the mock (GPU) device can load, to exercise
    // the diagnostic for a kernel left without any image.
    generateIgcaModule("IgcaOtherBackendKernel", "igca_10",
                       /*RecordTargets=*/true,
                       __SYCL_DEVICE_BINARY_TARGET_SPIRV64_X86_64)};

static unittest::MockDeviceImageArray<std::size(Imgs)> ImgArray{Imgs};

namespace {

std::vector<std::string> BinariesGivenToBackend;
std::vector<std::string> ILGivenToBackend;
// Largest number of images the runtime asked the backend to choose between.
// Above one means the runtime ranked images rather than picking one itself.
uint32_t MaxImagesOfferedToRanking = 0;
bool FailProgramBuild = false;

ur_result_t afterProgramCreateWithBinary(void *pParams) {
  auto Params = *static_cast<ur_program_create_with_binary_params_t *>(pParams);
  for (uint32_t I = 0; I < *Params.pnumDevices; ++I)
    BinariesGivenToBackend.push_back(
        reinterpret_cast<const char *>(*Params.pppBinaries[I]));
  return UR_RESULT_SUCCESS;
}

ur_result_t afterProgramCreateWithIL(void *pParams) {
  auto Params = *static_cast<ur_program_create_with_il_params_t *>(pParams);
  ILGivenToBackend.push_back(reinterpret_cast<const char *>(*Params.ppIL));
  return UR_RESULT_SUCCESS;
}

ur_result_t afterDeviceSelectBinary(void *pParams) {
  auto Params = *static_cast<ur_device_select_binary_params_t *>(pParams);
  MaxImagesOfferedToRanking =
      std::max(MaxImagesOfferedToRanking, *Params.pNumBinaries);
  **Params.ppSelectedBinary = 0;
  return UR_RESULT_SUCCESS;
}

ur_result_t replaceProgramBuildExp(void *) {
  return FailProgramBuild ? UR_RESULT_ERROR_INVALID_BINARY : UR_RESULT_SUCCESS;
}

// The mock device reports IGCA target 20 with the compute feature set, so a
// module holding only igca_20 code suits it and the diagnostics have a stable
// device IGCA target to name.
ur_result_t afterDeviceGetInfo(void *pParams) {
  auto Params = *static_cast<ur_device_get_info_params_t *>(pParams);
  const ur_device_info_t Name = *Params.ppropName;
  if (Name == UR_DEVICE_INFO_IGCA_TARGET) {
    if (*Params.ppPropValue)
      *static_cast<uint32_t *>(*Params.ppPropValue) = 20;
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = sizeof(uint32_t);
  } else if (Name == UR_DEVICE_INFO_IGCA_FEATURE_SET) {
    if (*Params.ppPropValue)
      *static_cast<ur_device_igca_feature_flags_t *>(*Params.ppPropValue) =
          UR_DEVICE_IGCA_FEATURE_FLAG_COMPUTE;
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = sizeof(ur_device_igca_feature_flags_t);
  }
  return UR_RESULT_SUCCESS;
}

class IgcaModuleSelectionTest : public ::testing::Test {
protected:
  void SetUp() override {
    BinariesGivenToBackend.clear();
    ILGivenToBackend.clear();
    MaxImagesOfferedToRanking = 0;
    FailProgramBuild = false;

    mock::getCallbacks().set_after_callback("urProgramCreateWithBinary",
                                            &afterProgramCreateWithBinary);
    mock::getCallbacks().set_after_callback("urProgramCreateWithIL",
                                            &afterProgramCreateWithIL);
    mock::getCallbacks().set_after_callback("urDeviceSelectBinary",
                                            &afterDeviceSelectBinary);
    mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                            &afterDeviceGetInfo);
    mock::getCallbacks().set_replace_callback("urProgramBuildExp",
                                              &replaceProgramBuildExp);
  }

  unittest::UrMock<> Mock;
};

} // namespace

// An uber-module's compile_target describes at most one of its sub-modules, so
// it must not be used to reject the module: the backend picks the sub-module.
TEST_F(IgcaModuleSelectionTest, IgcaModuleUsedDespiteCompileTarget) {
  queue{}.single_task<IgcaOnlyKernel>([]() {});

  ASSERT_EQ(BinariesGivenToBackend.size(), 1U);
  EXPECT_EQ(BinariesGivenToBackend.back(), "IgcaOnlyKernel");
}

// The --offload-arch module is more specific than a JIT module defining the
// same kernel, so it wins and the backend is never asked to rank the two.
TEST_F(IgcaModuleSelectionTest, IgcaModuleWinsWithoutRanking) {
  queue{}.single_task<IgcaAndJitKernel>([]() {});

  EXPECT_TRUE(ILGivenToBackend.empty());
  ASSERT_EQ(BinariesGivenToBackend.size(), 1U);
  EXPECT_EQ(BinariesGivenToBackend.back(), "IgcaAndJitKernel");
  EXPECT_LE(MaxImagesOfferedToRanking, 1U);
}

// The kernel_bundle queries have to find a kernel's defining module the same
// way kernel submission does, so they must not reject an --offload-arch module
// over its compile_target either.
TEST_F(IgcaModuleSelectionTest, KernelBundleFindsIgcaModule) {
  device Dev = platform{}.get_devices()[0];
  EXPECT_TRUE(sycl::is_compatible<IgcaOnlyKernel>(Dev));

  kernel_id Id = sycl::get_kernel_id<IgcaOnlyKernel>();
  kernel_bundle KB = sycl::get_kernel_bundle<bundle_state::executable>(
      context{Dev}, {Dev}, {Id});
  EXPECT_TRUE(KB.has_kernel(Id));
}

// Whether the new selection applies is a per-module decision: a kernel that is
// not in an --offload-arch module keeps being ranked by the backend, even
// though the same binary holds --offload-arch modules for other kernels.
TEST_F(IgcaModuleSelectionTest, NonIgcaKernelStillRanked) {
  queue{}.single_task<JitOnlyKernel>([]() {});

  EXPECT_GT(MaxImagesOfferedToRanking, 1U);
}

// A module the backend refuses must not reach the user as a bare backend
// error: there is no other module to try, so the diagnostic has to name the
// device and the IGCA targets the module holds.
TEST_F(IgcaModuleSelectionTest, BuildFailureNamesDeviceAndIgcaTargets) {
  FailProgramBuild = true;
  try {
    queue{}.single_task<IgcaBuildFailureKernel>([]() {});
    FAIL() << "Expected an exception from the failed module build";
  } catch (sycl::exception &E) {
    std::string What{E.what()};
    EXPECT_NE(What.find("--offload-arch"), std::string::npos) << What;
    EXPECT_NE(What.find("'Mock device'"), std::string::npos) << What;
    EXPECT_NE(What.find("IGCA target 20 (compute)"), std::string::npos) << What;
    EXPECT_NE(What.find("[igca_20]"), std::string::npos) << What;
  }
}

// The target list is optional, so a module that does not record it still gets
// the rest of the diagnostic.
TEST_F(IgcaModuleSelectionTest, BuildFailureWithoutRecordedIgcaTargets) {
  FailProgramBuild = true;
  try {
    queue{}.single_task<IgcaNoRecordedTargetsKernel>([]() {});
    FAIL() << "Expected an exception from the failed module build";
  } catch (sycl::exception &E) {
    std::string What{E.what()};
    EXPECT_NE(What.find("'Mock device'"), std::string::npos) << What;
    EXPECT_NE(What.find("does not record which IGCA targets"),
              std::string::npos)
        << What;
  }
}

// When an --offload-arch module is among a kernel's images, the device's IGCA
// target is what decides whether the module can run, so the diagnostic for a
// kernel left without any image names it along with the module's targets.
TEST_F(IgcaModuleSelectionTest, NoImageNamesDeviceIgcaTarget) {
  try {
    queue{}.single_task<IgcaOtherBackendKernel>([]() {});
    FAIL() << "Expected an exception for a kernel with no image for the device";
  } catch (sycl::exception &E) {
    EXPECT_EQ(std::string{E.what()},
              "Kernel IgcaOtherBackendKernel has no image for the selected "
              "device 'Mock device', IGCA target 20 (compute). Its available "
              "images target: [--offload-arch igca_10].");
  }
}
