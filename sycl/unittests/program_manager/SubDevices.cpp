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

static ur_device_handle_t rootDevice = (ur_device_handle_t)0x1;
// Sub-devices under rootDevice
static ur_device_handle_t urSubDev1 = (ur_device_handle_t)0x11;
static ur_device_handle_t urSubDev2 = (ur_device_handle_t)0x12;
// Sub-sub-devices under urSubDev1
static ur_device_handle_t urSubSubDev1 = (ur_device_handle_t)0x111;
static ur_device_handle_t urSubSubDev2 = (ur_device_handle_t)0x112;

namespace {
ur_result_t redefinedDeviceGet(void *pParams) {
  auto params = *static_cast<ur_device_get_params_t *>(pParams);
  if (*params.ppNumDevices)
    **params.ppNumDevices = 1;
  if (*params.pphDevices)
    (*params.pphDevices)[0] = rootDevice;
  return UR_RESULT_SUCCESS;
}

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
    if (!*params.ppPropValue)
      **params.ppPropSizeRet = sizeof(uint32_t);
    else
      ((uint32_t *)*params.ppPropValue)[0] = 2;
  }
  if (*params.ppropName == UR_DEVICE_INFO_PARENT_DEVICE) {
    if (!*params.ppPropValue) {
      **params.ppPropSizeRet = sizeof(ur_device_handle_t);
    } else {
      ur_device_handle_t &ret =
          *static_cast<ur_device_handle_t *>(*params.ppPropValue);
      if (*params.phDevice == urSubDev1 || *params.phDevice == urSubDev2) {
        ret = rootDevice;
      } else if (*params.phDevice == urSubSubDev1 ||
                 *params.phDevice == urSubSubDev2) {
        ret = urSubDev1;
      } else {
        ret = nullptr;
      }
    }
  }
  if (*params.ppropName == UR_DEVICE_INFO_BUILD_ON_SUBDEVICE) {
    if (!*params.ppPropValue)
      **params.ppPropSizeRet = sizeof(ur_bool_t);
    else
      ((ur_bool_t *)*params.ppPropValue)[0] = false;
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

static int buildCallCount = 0;

ur_result_t redefinedProgramBuildExp(void *) {
  buildCallCount++;
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
  sycl::detail::platform_impl &PltImpl = *sycl::detail::getSyclObjImpl(Plt);
  sycl::detail::device_impl &subDev1 = PltImpl.getOrMakeDeviceImpl(urSubDev1);
  sycl::detail::device_impl &subDev2 = PltImpl.getOrMakeDeviceImpl(urSubDev2);
  sycl::context Ctx{
      {device, sycl::detail::createSyclObjFromImpl<sycl::device>(subDev1),
       sycl::detail::createSyclObjFromImpl<sycl::device>(subDev2)}};

  // Create device binary description structures for getBuiltPIProgram API.
  auto devBin = Imgs[0].convertToNativeType();
  sycl_device_binaries_struct devBinStruct{SYCL_DEVICE_BINARIES_VERSION, 1,
                                           &devBin, nullptr, nullptr};
  sycl::detail::ProgramManager::getInstance().addImages(&devBinStruct);

  // Build program via getBuiltPIProgram API
  sycl::detail::ProgramManager::getInstance().getBuiltURProgram(
      *sycl::detail::getSyclObjImpl(Ctx), subDev1,
      sycl::detail::KernelInfo<TestKernel>::getName());
  // This call should re-use built binary from the cache. If urProgramBuild is
  // called again, the test will fail as second call of redefinedProgramBuild
  sycl::detail::ProgramManager::getInstance().getBuiltURProgram(
      *sycl::detail::getSyclObjImpl(Ctx), subDev2,
      sycl::detail::KernelInfo<TestKernel>::getName());
}

// Check that program is built once for all sub-sub-devices
TEST(SubDevices, BuildProgramForSubSubDevices) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urDeviceGet", &redefinedDeviceGet);
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfo);
  mock::getCallbacks().set_after_callback("urProgramBuildExp",
                                          &redefinedProgramBuildExp);
  sycl::platform Plt = sycl::platform();
  sycl::device root = Plt.get_devices()[0];
  sycl::detail::platform_impl &PltImpl = *sycl::detail::getSyclObjImpl(Plt);
  // Initialize sub-sub-devices
  sycl::detail::device_impl &SubSub1 =
      PltImpl.getOrMakeDeviceImpl(urSubSubDev1);
  sycl::detail::device_impl &SubSub2 =
      PltImpl.getOrMakeDeviceImpl(urSubSubDev2);

  sycl::context Ctx{root};
  buildCallCount = 0;
  sycl::detail::ProgramManager::getInstance().getBuiltURProgram(
      *sycl::detail::getSyclObjImpl(Ctx), SubSub1,
      sycl::detail::KernelInfo<TestKernel>::getName());
  sycl::detail::ProgramManager::getInstance().getBuiltURProgram(
      *sycl::detail::getSyclObjImpl(Ctx), SubSub2,
      sycl::detail::KernelInfo<TestKernel>::getName());

  // Check that program is built only once.
  EXPECT_EQ(buildCallCount, 1);
}
