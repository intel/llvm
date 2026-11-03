//==----------------------- KernelAttributes.cpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Checks that info::kernel::attributes is built from the device requirements
// of the image when the backend reports no attributes.
//
//===----------------------------------------------------------------------===//

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

class KernelAttributesTestKernel;
MOCK_INTEGRATION_HEADER(KernelAttributesTestKernel)

static sycl::unittest::MockDeviceImage generateImage() {
  using namespace sycl::unittest;

  std::vector<char> SubGroupSize(sizeof(uint32_t));
  SubGroupSize[0] = 8;

  MockPropertySet PropSet;
  PropSet.insert(__SYCL_PROPERTY_SET_SYCL_DEVICE_REQUIREMENTS,
                 std::vector<MockProperty>{
                     makeAspectsProp({sycl::aspect::gpu}),
                     makeReqdWGSizeProp({16, 4}),
                     {"reqd_sub_group_size", SubGroupSize,
                      SYCL_PROPERTY_TYPE_UINT32}});
  return MockDeviceImage(makeEmptyKernels({"KernelAttributesTestKernel"}),
                         std::move(PropSet));
}

static sycl::unittest::MockDeviceImage Imgs[] = {generateImage()};
static sycl::unittest::MockDeviceImageArray<1> ImgArray{Imgs};

static ur_result_t afterKernelGetInfo(void *pParams) {
  auto params = *static_cast<ur_kernel_get_info_params_t *>(pParams);
  constexpr char KernelName[] = "KernelAttributesTestKernel";
  if (*params.ppropName == UR_KERNEL_INFO_FUNCTION_NAME) {
    if (*params.ppPropValue)
      std::memcpy(*params.ppPropValue, KernelName, sizeof(KernelName));
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(KernelName);
  } else if (*params.ppropName == UR_KERNEL_INFO_ATTRIBUTES) {
    // The backend reports no attributes.
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = 0;
  }
  return UR_RESULT_SUCCESS;
}

static ur_result_t afterDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  // Let the image pass the device requirement checks of the executable bundle.
  if (*params.ppropName == UR_DEVICE_INFO_MAX_WORK_GROUP_SIZE) {
    if (*params.ppPropValue)
      *static_cast<size_t *>(*params.ppPropValue) = 256;
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(size_t);
  } else if (*params.ppropName == UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL) {
    if (*params.ppPropValue)
      *static_cast<uint32_t *>(*params.ppPropValue) = 8;
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(uint32_t);
  }
  return UR_RESULT_SUCCESS;
}

TEST(KernelAttributes, FallbackToDeviceRequirements) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urKernelGetInfo",
                                          &afterKernelGetInfo);
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &afterDeviceGetInfo);

  try {
    sycl::context Ctx{sycl::platform().get_devices()[0]};
    auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx);
    sycl::kernel Kernel = KB.get_kernel<KernelAttributesTestKernel>();

    EXPECT_EQ(Kernel.get_info<sycl::info::kernel::attributes>(),
              "reqd_work_group_size(4, 16, 1) reqd_sub_group_size(8) "
              "device_has(aspect::gpu)");
  } catch (const sycl::exception &E) {
    FAIL() << "Unexpected exception: " << E.what();
  }
}
