//==---- KernelBundleStateFiltering.cpp --- Kernel bundle unit test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_impl.hpp>
#include <detail/kernel_bundle_impl.hpp>
#include <sycl/sycl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <vector>

class KernelA;
class KernelB;
class KernelC;
class KernelD;
class KernelE;

MOCK_INTEGRATION_HEADER(KernelA)
MOCK_INTEGRATION_HEADER(KernelB)
MOCK_INTEGRATION_HEADER(KernelC)
MOCK_INTEGRATION_HEADER(KernelD)
MOCK_INTEGRATION_HEADER(KernelE)

namespace {

std::set<const void *> TrackedImages;
sycl::unittest::MockDeviceImage
generateDefaultImage(std::initializer_list<std::string> KernelNames,
                     sycl_device_binary_type BinaryType,
                     const char *DeviceTargetSpec) {
  using namespace sycl::unittest;

  MockPropertySet PropSet;

  static unsigned char NImage = 0;
  std::vector<unsigned char> Bin{NImage++};

  std::vector<MockOffloadEntry> Entries = makeEmptyKernels(KernelNames);

  MockDeviceImage Img{BinaryType, // Format
                      DeviceTargetSpec,
                      "", // Compile options
                      "", // Link options
                      std::move(Bin),
                      std::move(Entries),
                      std::move(PropSet)};
  const void *BinaryPtr = Img.getBinaryPtr();
  TrackedImages.insert(BinaryPtr);

  return Img;
}

// Image 0: input, KernelA KernelB
// Image 1: exe, KernelA
// Image 2: input, KernelC
// Image 3: exe, KernelC
// Image 4: input, KernelD
// Image 5: input, KernelE
// Image 6: exe, KernelE
// Image 7: exe. KernelE
sycl::unittest::MockDeviceImage Imgs[] = {
    generateDefaultImage({"KernelA", "KernelB"}, SYCL_DEVICE_BINARY_TYPE_SPIRV,
                         __SYCL_DEVICE_BINARY_TARGET_SPIRV64),
    generateDefaultImage({"KernelA"}, SYCL_DEVICE_BINARY_TYPE_NATIVE,
                         __SYCL_DEVICE_BINARY_TARGET_SPIRV64_X86_64),
    generateDefaultImage({"KernelC"}, SYCL_DEVICE_BINARY_TYPE_SPIRV,
                         __SYCL_DEVICE_BINARY_TARGET_SPIRV64),
    generateDefaultImage({"KernelC"}, SYCL_DEVICE_BINARY_TYPE_NATIVE,
                         __SYCL_DEVICE_BINARY_TARGET_SPIRV64_FPGA),
    generateDefaultImage({"KernelD"}, SYCL_DEVICE_BINARY_TYPE_SPIRV,
                         __SYCL_DEVICE_BINARY_TARGET_SPIRV64),
    generateDefaultImage({"KernelE"}, SYCL_DEVICE_BINARY_TYPE_SPIRV,
                         __SYCL_DEVICE_BINARY_TARGET_SPIRV64),
    generateDefaultImage({"KernelE"}, SYCL_DEVICE_BINARY_TYPE_NATIVE,
                         __SYCL_DEVICE_BINARY_TARGET_SPIRV64_X86_64),
    generateDefaultImage({"KernelE"}, SYCL_DEVICE_BINARY_TYPE_NATIVE,
                         __SYCL_DEVICE_BINARY_TARGET_SPIRV64_X86_64)};

sycl::unittest::MockDeviceImageArray<std::size(Imgs)> ImgArray{Imgs};
std::vector<unsigned char> UsedImageIndices;

void redefinedUrProgramCreateCommon(const void *bin) {
  if (TrackedImages.count(bin) != 0) {
    unsigned char ImgIdx = *reinterpret_cast<const unsigned char *>(bin);
    UsedImageIndices.push_back(ImgIdx);
  }
}

ur_result_t redefinedUrProgramCreate(void *pParams) {
  auto params = *static_cast<ur_program_create_with_il_params_t *>(pParams);
  redefinedUrProgramCreateCommon(*params.ppIL);
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedUrProgramCreateWithBinary(void *pParams) {
  auto params = *static_cast<ur_program_create_with_binary_params_t *>(pParams);
  for (uint32_t i = 0; i < *params.pnumDevices; ++i)
    redefinedUrProgramCreateCommon(*params.pppBinaries[i]);
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedDevicesGet(void *pParams) {
  auto params = *static_cast<ur_device_get_params_t *>(pParams);
  if (*params.ppNumDevices)
    **params.ppNumDevices = 2;

  if (*params.pphDevices) {
    (*params.pphDevices)[0] = reinterpret_cast<ur_device_handle_t>(1);
    (*params.pphDevices)[1] = reinterpret_cast<ur_device_handle_t>(2);
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedDeviceSelectBinary(void *pParams) {
  auto params = *static_cast<ur_device_select_binary_params_t *>(pParams);
  EXPECT_EQ(*params.pNumBinaries, 1U);
  // Treat image 3 as incompatible with one of the devices.
  //
  // FIXME: this is expecting ur_device_binary so it can do stuff with the
  // actual binary, not just the metadata.. not sure how we're going to support
  // this
  std::string BinarySpec = (*params.ppBinaries)[0].pDeviceTargetSpec;
  if (BinarySpec.find("spir64_fpga") != std::string::npos &&
      *params.phDevice == reinterpret_cast<ur_device_handle_t>(2)) {
    return UR_RESULT_ERROR_INVALID_BINARY;
  }
  **params.ppSelectedBinary = 0;
  return UR_RESULT_SUCCESS;
}

void verifyImageUse(const std::vector<unsigned char> &ExpectedImages) {
  std::sort(UsedImageIndices.begin(), UsedImageIndices.end());
  EXPECT_TRUE(std::is_sorted(ExpectedImages.begin(), ExpectedImages.end()));
  EXPECT_EQ(UsedImageIndices, ExpectedImages);
  if (UsedImageIndices != ExpectedImages) {
    printf("break here\n");
  }
  UsedImageIndices.clear();
}

TEST(KernelBundle, DeviceImageStateFiltering) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urProgramCreateWithIL",
                                          &redefinedUrProgramCreate);
  mock::getCallbacks().set_after_callback("urProgramCreateWithBinary",
                                          &redefinedUrProgramCreateWithBinary);
  // No kernel ids specified.
  {
    const sycl::device Dev = sycl::platform().get_devices()[0];
    sycl::context Ctx{Dev};

    sycl::kernel_bundle<sycl::bundle_state::executable> KernelBundle =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev});
    verifyImageUse({0, 1, 3, 4, 6, 7});
  }

  sycl::kernel_id KernelAID = sycl::get_kernel_id<KernelA>();
  sycl::kernel_id KernelCID = sycl::get_kernel_id<KernelC>();
  sycl::kernel_id KernelDID = sycl::get_kernel_id<KernelD>();

  // Request specific kernel ids.
  {
    const sycl::device Dev = sycl::platform().get_devices()[0];
    sycl::context Ctx{Dev};

    sycl::kernel_bundle<sycl::bundle_state::executable> KernelBundle =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            Ctx, {Dev}, {KernelAID, KernelCID, KernelDID});
    verifyImageUse({1, 3, 4});
  }

  // Check the case where some executable images are unsupported by one of
  // the devices.
  {
    mock::getCallbacks().set_replace_callback("urDeviceGet",
                                              &redefinedDevicesGet);
    mock::getCallbacks().set_replace_callback("urDeviceSelectBinary",
                                              &redefinedDeviceSelectBinary);
    const std::vector<sycl::device> Devs = sycl::platform().get_devices();
    sycl::context Ctx{Devs};

    sycl::kernel_bundle<sycl::bundle_state::executable> KernelBundle =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            Ctx, Devs, {KernelAID, KernelCID, KernelDID});
    verifyImageUse({1, 2, 3, 4});
  }
}
} // namespace
