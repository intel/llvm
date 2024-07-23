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

#include <helpers/MockKernelInfo.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

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
sycl::unittest::PiImage
generateDefaultImage(std::initializer_list<std::string> KernelNames,
                     pi_device_binary_type BinaryType,
                     const char *DeviceTargetSpec) {
  using namespace sycl::unittest;

  PiPropertySet PropSet;

  static unsigned char NImage = 0;
  std::vector<unsigned char> Bin{NImage++};

  PiArray<PiOffloadEntry> Entries = makeEmptyKernels(KernelNames);

  PiImage Img{BinaryType, // Format
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
sycl::unittest::PiImage Imgs[] = {
    generateDefaultImage({"KernelA", "KernelB"}, PI_DEVICE_BINARY_TYPE_SPIRV,
                         __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64),
    generateDefaultImage({"KernelA"}, PI_DEVICE_BINARY_TYPE_NATIVE,
                         __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64),
    generateDefaultImage({"KernelC"}, PI_DEVICE_BINARY_TYPE_SPIRV,
                         __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64),
    generateDefaultImage({"KernelC"}, PI_DEVICE_BINARY_TYPE_NATIVE,
                         __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64),
    generateDefaultImage({"KernelD"}, PI_DEVICE_BINARY_TYPE_SPIRV,
                         __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64),
    generateDefaultImage({"KernelE"}, PI_DEVICE_BINARY_TYPE_SPIRV,
                         __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64),
    generateDefaultImage({"KernelE"}, PI_DEVICE_BINARY_TYPE_NATIVE,
                         __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64),
    generateDefaultImage({"KernelE"}, PI_DEVICE_BINARY_TYPE_NATIVE,
                         __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64)};

sycl::unittest::PiImageArray<std::size(Imgs)> ImgArray{Imgs};
std::vector<unsigned char> UsedImageIndices;

void redefinedPiProgramCreateCommon(const void *bin) {
  if (TrackedImages.count(bin) != 0) {
    unsigned char ImgIdx = *reinterpret_cast<const unsigned char *>(bin);
    UsedImageIndices.push_back(ImgIdx);
  }
}

pi_result redefinedPiProgramCreate(pi_context context, const void *il,
                                   size_t length, pi_program *res_program) {
  redefinedPiProgramCreateCommon(il);
  return PI_SUCCESS;
}

pi_result redefinedPiProgramCreateWithBinary(
    pi_context context, pi_uint32 num_devices, const pi_device *device_list,
    const size_t *lengths, const unsigned char **binaries,
    size_t num_metadata_entries, const pi_device_binary_property *metadata,
    pi_int32 *binary_status, pi_program *ret_program) {
  redefinedPiProgramCreateCommon(binaries[0]);
  return PI_SUCCESS;
}

pi_result redefinedDevicesGet(pi_platform platform, pi_device_type device_type,
                              pi_uint32 num_entries, pi_device *devices,
                              pi_uint32 *num_devices) {
  if (num_devices)
    *num_devices = 2;

  if (devices) {
    devices[0] = reinterpret_cast<pi_device>(1);
    devices[1] = reinterpret_cast<pi_device>(2);
  }

  return PI_SUCCESS;
}

pi_result redefinedExtDeviceSelectBinary(pi_device device,
                                         pi_device_binary *binaries,
                                         pi_uint32 num_binaries,
                                         pi_uint32 *selected_binary_ind) {
  EXPECT_EQ(num_binaries, 1U);
  // Treat image 3 as incompatible with one of the devices.
  if (TrackedImages.count(binaries[0]->BinaryStart) != 0 &&
      *binaries[0]->BinaryStart == 3 &&
      device == reinterpret_cast<pi_device>(2)) {
    return PI_ERROR_INVALID_BINARY;
  }
  *selected_binary_ind = 0;
  return PI_SUCCESS;
}

void verifyImageUse(const std::vector<unsigned char> &ExpectedImages) {
  std::sort(UsedImageIndices.begin(), UsedImageIndices.end());
  EXPECT_TRUE(std::is_sorted(ExpectedImages.begin(), ExpectedImages.end()));
  EXPECT_EQ(UsedImageIndices, ExpectedImages);
  UsedImageIndices.clear();
}

TEST(KernelBundle, DeviceImageStateFiltering) {
  sycl::unittest::PiMock Mock;
  Mock.redefineAfter<sycl::detail::PiApiKind::piProgramCreate>(
      redefinedPiProgramCreate);
  Mock.redefineAfter<sycl::detail::PiApiKind::piProgramCreateWithBinary>(
      redefinedPiProgramCreateWithBinary);

  // No kernel ids specified.
  {
    const sycl::device Dev = Mock.getPlatform().get_devices()[0];
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
    const sycl::device Dev = Mock.getPlatform().get_devices()[0];
    sycl::context Ctx{Dev};

    sycl::kernel_bundle<sycl::bundle_state::executable> KernelBundle =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            Ctx, {Dev}, {KernelAID, KernelCID, KernelDID});
    verifyImageUse({1, 3, 4});
  }

  // Check the case where some executable images are unsupported by one of
  // the devices.
  {
    Mock.redefine<sycl::detail::PiApiKind::piDevicesGet>(redefinedDevicesGet);
    Mock.redefine<sycl::detail::PiApiKind::piextDeviceSelectBinary>(
        redefinedExtDeviceSelectBinary);
    const std::vector<sycl::device> Devs = Mock.getPlatform().get_devices();
    sycl::context Ctx{Devs};

    sycl::kernel_bundle<sycl::bundle_state::executable> KernelBundle =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            Ctx, Devs, {KernelAID, KernelCID, KernelDID});
    verifyImageUse({1, 2, 3, 4});
  }
}
} // namespace
