#include <sycl/sycl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

// Check that an AOT image built for one GPU architecture only makes devices of
// that architecture preferred by the default device selector.

class CompileTargetSelectionKernel;

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<CompileTargetSelectionKernel>
    : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() {
    return "CompileTargetSelectionKernel";
  }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

namespace {
namespace exp = sycl::ext::oneapi::experimental;

sycl::unittest::MockDeviceImage generatePVCImage() {
  std::string CompileTarget = "intel_gpu_pvc";
  std::vector<char> Data(8 + CompileTarget.size());
  std::copy(CompileTarget.begin(), CompileTarget.end(), Data.data() + 8);
  sycl::unittest::MockPropertySet PropSet;
  PropSet.insert(__SYCL_PROPERTY_SET_SYCL_DEVICE_REQUIREMENTS,
                 sycl::unittest::MockProperty("compile_target", Data,
                                              SYCL_PROPERTY_TYPE_BYTE_ARRAY));
  return sycl::unittest::MockDeviceImage{
      SYCL_DEVICE_BINARY_TYPE_NATIVE,
      __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN,
      "", // Compile options
      "", // Link options
      std::vector<unsigned char>{1},
      sycl::unittest::makeEmptyKernels({"CompileTargetSelectionKernel"}),
      std::move(PropSet)};
}
sycl::unittest::MockDeviceImage Imgs[] = {generatePVCImage()};
sycl::unittest::MockDeviceImageArray<1> ImgArray{Imgs};

// IP versions of a TGLLP and a PVC GPU. The TGLLP one comes first so that it
// would be chosen if both devices were preferred equally.
uint32_t MockDeviceIPs[] = {0x03000000, 0x030f0000};

ur_result_t redefinedDeviceGet(void *pParams) {
  auto Params = *static_cast<ur_device_get_params_t *>(pParams);
  if (*Params.ppNumDevices) {
    **Params.ppNumDevices = std::size(MockDeviceIPs);
    return UR_RESULT_SUCCESS;
  }
  if (*Params.pphDevices)
    for (uint32_t I = 0; I < *Params.pNumEntries; ++I)
      (*Params.pphDevices)[I] =
          reinterpret_cast<ur_device_handle_t>(&MockDeviceIPs[I]);
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedDeviceGetInfo(void *pParams) {
  auto Params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*Params.ppropName == UR_DEVICE_INFO_IP_VERSION && *Params.ppPropValue)
    *static_cast<uint32_t *>(*Params.ppPropValue) =
        *reinterpret_cast<uint32_t *>(*Params.phDevice);
  if (*Params.ppropName == UR_DEVICE_INFO_TYPE && *Params.ppPropValue)
    *static_cast<ur_device_type_t *>(*Params.ppPropValue) = UR_DEVICE_TYPE_GPU;
  return UR_RESULT_SUCCESS;
}
} // namespace

TEST(CompileTargetSelection, DefaultSelectorPrefersDeviceWithImage) {
  sycl::unittest::UrMock<sycl::backend::ext_oneapi_level_zero> Mock;
  mock::getCallbacks().set_replace_callback("urDeviceGet", &redefinedDeviceGet);
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfo);

  std::vector<sycl::device> Devs = sycl::platform().get_devices();
  ASSERT_EQ(Devs.size(), 2u);
  ASSERT_EQ(Devs[0].get_info<exp::info::device::architecture>(),
            exp::architecture::intel_gpu_tgllp);
  ASSERT_EQ(Devs[1].get_info<exp::info::device::architecture>(),
            exp::architecture::intel_gpu_pvc);
  EXPECT_LT(sycl::default_selector_v(Devs[0]),
            sycl::default_selector_v(Devs[1]));

  sycl::device Dev{sycl::default_selector_v};
  EXPECT_EQ(Dev.get_info<exp::info::device::architecture>(),
            exp::architecture::intel_gpu_pvc);
}
