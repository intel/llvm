//==-------------------------- KernelInfoShortcuts.cpp -------------------==//
//
// Unit test to ensure get_kernel_info for a device queries/uses kernel bundle
// for that specific device only and doesn't trigger builds for all devices.
//

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

using namespace sycl;
using namespace sycl::unittest;

class ShortcutKernelInfoTestKernel;
MOCK_INTEGRATION_HEADER(ShortcutKernelInfoTestKernel)

static int ProgramBuildCounter = 0;
static ur_result_t redefinedurProgramBuild(void *pParams) {
  ++ProgramBuildCounter;
  return UR_RESULT_SUCCESS;
}

static ur_result_t redefinedDeviceGet(void *pParams) {
  auto params = *static_cast<ur_device_get_params_t *>(pParams);
  if (*params.ppNumDevices) {
    **params.ppNumDevices = 2; // two devices total
    return UR_RESULT_SUCCESS;
  }
  if (*params.pphDevices) {
    // provide two mock device handles
    (*params.pphDevices)[0] = reinterpret_cast<ur_device_handle_t>(0x1);
    (*params.pphDevices)[1] = reinterpret_cast<ur_device_handle_t>(0x2);
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedurKernelGetGroupInfo(void *pParams) {
  return UR_RESULT_SUCCESS;
}

TEST(ShortcutKernelInfo, QueryInfoForSingleDevice) {
  unittest::UrMock<> Mock;
  static sycl::unittest::MockDeviceImage DevImage =
      sycl::unittest::generateDefaultImage({"ShortcutKernelInfoTestKernel"});
  static sycl::unittest::MockDeviceImageArray<1> DevImageArray = {&DevImage};

  mock::getCallbacks().set_replace_callback("urDeviceGet", &redefinedDeviceGet);
  mock::getCallbacks().set_replace_callback("urProgramBuildExp",
                                            &redefinedurProgramBuild);
  mock::getCallbacks().set_replace_callback("urKernelGetGroupInfo",
                                            &redefinedurKernelGetGroupInfo);

  platform Plt = platform();
  std::vector<device> Devs = Plt.get_devices();
  ASSERT_GE(Devs.size(), 2u) << "Test requires at least 2 devices";
  context Ctx = context(Devs);
  queue Queue = queue(Ctx, Devs[0]);

  // Query kernel info for the first device only
  ProgramBuildCounter = 0;
  sycl::ext::oneapi::get_kernel_info<
      ShortcutKernelInfoTestKernel,
      sycl::info::kernel_device_specific::work_group_size>(Ctx, Devs[0]);
  sycl::ext::oneapi::get_kernel_info<
      ShortcutKernelInfoTestKernel,
      sycl::info::kernel_device_specific::work_group_size>(Queue);

  EXPECT_EQ(ProgramBuildCounter, 1);
}
