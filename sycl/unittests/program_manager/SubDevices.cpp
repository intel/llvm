//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/program.hpp>
#include <detail/kernel_bundle_impl.hpp>

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

#include <helpers/TestKernel.hpp>

static pi_device rootDevice;
static pi_device piSubDev1 = (pi_device)0x1;
static pi_device piSubDev2 = (pi_device)0x2;

namespace {
pi_result redefinedDeviceGetInfo(pi_device device, pi_device_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t *param_value_size_ret) {
  if (param_name == PI_DEVICE_INFO_PARTITION_PROPERTIES) {
    if (!param_value) {
      *param_value_size_ret = 2 * sizeof(pi_device_partition_property);
    } else {
      ((pi_device_partition_property *)param_value)[0] =
          PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
      ((pi_device_partition_property *)param_value)[1] =
          PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
    }
  }
  if (param_name == PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN) {
    if (!param_value) {
      *param_value_size_ret = sizeof(pi_device_affinity_domain);
    } else {
      ((pi_device_affinity_domain *)param_value)[0] =
          PI_DEVICE_AFFINITY_DOMAIN_NUMA |
          PI_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE;
    }
  }
  if (param_name == PI_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES) {
    ((pi_uint32 *)param_value)[0] = 2;
  }
  if (param_name == PI_DEVICE_INFO_PARENT_DEVICE) {
    if (device == piSubDev1 || device == piSubDev2)
      ((pi_device *)param_value)[0] = rootDevice;
    else
      ((pi_device *)param_value)[0] = nullptr;
  }
  return PI_SUCCESS;
}

pi_result redefinedDevicePartition(
    pi_device Device, const pi_device_partition_property *Properties,
    pi_uint32 NumDevices, pi_device *OutDevices, pi_uint32 *OutNumDevices) {
  if (OutNumDevices)
    *OutNumDevices = 2;
  if (OutDevices) {
    OutDevices[0] = {};
    OutDevices[1] = {};
  }
  return PI_SUCCESS;
}

pi_result redefinedDeviceRetain(pi_device c) { return PI_SUCCESS; }

pi_result redefinedDeviceRelease(pi_device c) { return PI_SUCCESS; }

pi_result redefinedProgramBuild(
    pi_program prog, pi_uint32, const pi_device *, const char *,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data) {
  static int m = 0;
  m++;
  // if called more than once return an error
  if (m > 1)
    return PI_ERROR_UNKNOWN;

  return PI_SUCCESS;
}

pi_result redefinedContextCreate(const pi_context_properties *Properties,
                                 pi_uint32 NumDevices, const pi_device *Devices,
                                 void (*PFnNotify)(const char *ErrInfo,
                                                   const void *PrivateInfo,
                                                   size_t CB, void *UserData),
                                 void *UserData, pi_context *RetContext) {
  return PI_SUCCESS;
}
} // anonymous namespace

// Check that program is built once for all sub-devices
// FIXME: mock 3 devices (one root device + two sub-devices) within a single
// context.
TEST(SubDevices, DISABLED_BuildProgramForSubdevices) {
  sycl::platform Plt{sycl::default_selector()};
  // Host devices do not support sub-devices
  if (Plt.is_host() || Plt.get_backend() == sycl::backend::ext_oneapi_cuda ||
      Plt.get_backend() == sycl::backend::ext_oneapi_hip) {
    std::cerr << "Test is not supported on "
              << Plt.get_info<sycl::info::platform::name>() << ", skipping\n";
    GTEST_SKIP(); // test is not supported on selected platform.
  }

  // Setup Mock APIs
  sycl::unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  Mock.redefine<sycl::detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfo);
  Mock.redefine<sycl::detail::PiApiKind::piDevicePartition>(
      redefinedDevicePartition);
  Mock.redefine<sycl::detail::PiApiKind::piDeviceRetain>(redefinedDeviceRetain);
  Mock.redefine<sycl::detail::PiApiKind::piDeviceRelease>(
      redefinedDeviceRelease);
  Mock.redefine<sycl::detail::PiApiKind::piProgramBuild>(redefinedProgramBuild);
  Mock.redefine<sycl::detail::PiApiKind::piContextCreate>(
      redefinedContextCreate);

  // Create 2 sub-devices and use first platform device as a root device
  const sycl::device device = Plt.get_devices()[0];
  // Initialize root device
  rootDevice = sycl::detail::getSyclObjImpl(device)->getHandleRef();
  // Initialize sub-devices
  auto PltImpl = sycl::detail::getSyclObjImpl(Plt);
  auto subDev1 =
      std::make_shared<sycl::detail::device_impl>(piSubDev1, PltImpl);
  auto subDev2 =
      std::make_shared<sycl::detail::device_impl>(piSubDev2, PltImpl);
  sycl::context Ctx{
      {device, sycl::detail::createSyclObjFromImpl<sycl::device>(subDev1),
       sycl::detail::createSyclObjFromImpl<sycl::device>(subDev2)}};

  // Create device binary description structures for getBuiltPIProgram API.
  auto devBin = Img.convertToNativeType();
  pi_device_binaries_struct devBinStruct{PI_DEVICE_BINARIES_VERSION, 1,
                                         &devBin, nullptr, nullptr};
  sycl::detail::ProgramManager::getInstance().addImages(&devBinStruct);

  // Build program via getBuiltPIProgram API
  sycl::detail::ProgramManager::getInstance().getBuiltPIProgram(
      sycl::detail::OSUtil::getOSModuleHandle(&devBin),
      sycl::detail::getSyclObjImpl(Ctx), subDev1,
      sycl::detail::KernelInfo<TestKernel>::getName());
  // This call should re-use built binary from the cache. If piProgramBuild is
  // called again, the test will fail as second call of redefinedProgramBuild
  sycl::detail::ProgramManager::getInstance().getBuiltPIProgram(
      sycl::detail::OSUtil::getOSModuleHandle(&devBin),
      sycl::detail::getSyclObjImpl(Ctx), subDev2,
      sycl::detail::KernelInfo<TestKernel>::getName());
}
