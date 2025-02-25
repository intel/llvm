//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/kernel_bundle_impl.hpp>

#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <helpers/TestKernel.hpp>

static ur_device_handle_t rootDevice;
static ur_device_handle_t urSubDev1 = (ur_device_handle_t)0x1;
static ur_device_handle_t urSubDev2 = (ur_device_handle_t)0x2;

namespace {
ur_result_t redefinedDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_SUPPORTED_PARTITIONS) {
    if (!*params.ppPropValue) {
      **params.ppPropSizeRet = 2 * sizeof(ur_device_partition_t);
    } else {
      ((ur_device_partition_t *)*params.ppPropValue)[0] =
          UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
      ((ur_device_partition_t *)*params.ppPropValue)[1] =
          UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
    }
  }
  if (*params.ppropName == UR_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN) {
    if (!*params.ppPropValue) {
      **params.ppPropSizeRet = sizeof(ur_device_affinity_domain_flags_t);
    } else {
      ((ur_device_affinity_domain_flags_t *)*params.ppPropValue)[0] =
          UR_DEVICE_AFFINITY_DOMAIN_FLAG_NUMA |
          UR_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE;
    }
  }
  if (*params.ppropName == UR_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES) {
    ((uint32_t *)*params.ppPropValue)[0] = 2;
  }
  if (*params.ppropName == UR_DEVICE_INFO_PARENT_DEVICE) {
    if (*params.phDevice == urSubDev1 || *params.phDevice == urSubDev2)
      ((ur_device_handle_t *)*params.ppPropValue)[0] = rootDevice;
    else
      ((ur_device_handle_t *)*params.ppPropValue)[0] = nullptr;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedDevicePartition(void *pParams) {
  auto params = *static_cast<ur_device_partition_params_t *>(pParams);
  if (*params.ppNumDevicesRet)
    **params.ppNumDevicesRet = 2;
  if (*params.pphSubDevices) {
    (*params.pphSubDevices)[0] = {};
    (*params.pphSubDevices)[1] = {};
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedDeviceRetain(void *) { return UR_RESULT_SUCCESS; }

ur_result_t redefinedDeviceRelease(void *) { return UR_RESULT_SUCCESS; }

ur_result_t redefinedProgramBuild(void *) {
  static int m = 0;
  m++;
  // if called more than once return an error
  if (m > 1)
    return UR_RESULT_ERROR_UNKNOWN;

  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedContextCreate(void *) { return UR_RESULT_SUCCESS; }
} // anonymous namespace

// Check that program is built once for all sub-devices
// FIXME: mock 3 devices (one root device + two sub-devices) within a single
// context.
TEST(SubDevices, DISABLED_BuildProgramForSubdevices) {
  // Setup Mock APIs
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_before_callback("urDeviceGetInfo",
                                           &redefinedDeviceGetInfo);
  mock::getCallbacks().set_before_callback("urDevicePartition",
                                           &redefinedDevicePartition);
  mock::getCallbacks().set_before_callback("urDeviceRetain",
                                           &redefinedDeviceRetain);
  mock::getCallbacks().set_before_callback("urDeviceRelease",
                                           &redefinedDeviceRelease);
  mock::getCallbacks().set_before_callback("urProgramBuild",
                                           &redefinedProgramBuild);
  mock::getCallbacks().set_before_callback("urContextCreate",
                                           &redefinedContextCreate);

  // Create 2 sub-devices and use first platform device as a root device
  const sycl::device device = Plt.get_devices()[0];
  // Initialize root device
  rootDevice = sycl::detail::getSyclObjImpl(device)->getHandleRef();
  // Initialize sub-devices
  auto PltImpl = sycl::detail::getSyclObjImpl(Plt);
  auto subDev1 =
      std::make_shared<sycl::detail::device_impl>(urSubDev1, PltImpl);
  auto subDev2 =
      std::make_shared<sycl::detail::device_impl>(urSubDev2, PltImpl);
  sycl::context Ctx{
      {device, sycl::detail::createSyclObjFromImpl<sycl::device>(subDev1),
       sycl::detail::createSyclObjFromImpl<sycl::device>(subDev2)}};

  // Create device binary description structures for getBuiltPIProgram API.
  auto devBin = Img.convertToNativeType();
  sycl_device_binaries_struct devBinStruct{SYCL_DEVICE_BINARIES_VERSION, 1,
                                           &devBin, nullptr, nullptr};
  sycl::detail::ProgramManager::getInstance().addImages(&devBinStruct);

  // Build program via getBuiltPIProgram API
  sycl::detail::ProgramManager::getInstance().getBuiltURProgram(
      sycl::detail::getSyclObjImpl(Ctx), subDev1,
      sycl::detail::KernelInfo<TestKernel<>>::getName());
  // This call should re-use built binary from the cache. If urProgramBuild is
  // called again, the test will fail as second call of redefinedProgramBuild
  sycl::detail::ProgramManager::getInstance().getBuiltURProgram(
      sycl::detail::getSyclObjImpl(Ctx), subDev2,
      sycl::detail::KernelInfo<TestKernel<>>::getName());
}
