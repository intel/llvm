//==----------------- DeviceCheck.cpp --- queue unit tests -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/config.hpp>
#include <detail/device_impl.hpp>
#include <gtest/gtest.h>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

namespace {

inline constexpr auto EnableDefaultContextsName =
    "SYCL_ENABLE_DEFAULT_CONTEXTS";

ur_device_handle_t ParentDevice = nullptr;
ur_platform_handle_t UrPlatform = nullptr;

ur_result_t redefinedDeviceGetInfoAfter(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_DEVICE_INFO_SUPPORTED_PARTITIONS) {
    if (*params.ppPropValue) {
      auto *Result =
          reinterpret_cast<ur_device_partition_t *>(*params.ppPropValue);
      *Result = UR_DEVICE_PARTITION_EQUALLY;
    }
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_device_partition_t);
  } else if (*params.ppropName == UR_DEVICE_INFO_MAX_COMPUTE_UNITS) {
    auto *Result = reinterpret_cast<uint32_t *>(*params.ppPropValue);
    *Result = 2;
  } else if (*params.ppropName == UR_DEVICE_INFO_PARENT_DEVICE) {
    auto *Result = reinterpret_cast<ur_device_handle_t *>(*params.ppPropValue);
    *Result = (*params.phDevice == ParentDevice) ? nullptr : ParentDevice;
  } else if (*params.ppropName == UR_DEVICE_INFO_PLATFORM) {
    auto *Result =
        reinterpret_cast<ur_platform_handle_t *>(*params.ppPropValue);
    *Result = UrPlatform;
  } else if (*params.ppropName == UR_DEVICE_INFO_EXTENSIONS) {
    if (*params.ppPropSizeRet) {
      **params.ppPropSizeRet = 0;
    }
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t redefinedDevicePartitionAfter(void *pParams) {
  auto params = *static_cast<ur_device_partition_params_t *>(pParams);
  if (*params.pphSubDevices) {
    for (size_t I = 0; I < *params.pNumDevices; ++I) {
      *params.pphSubDevices[I] = reinterpret_cast<ur_device_handle_t>(1000 + I);
    }
  }
  if (*params.ppNumDevicesRet)
    **params.ppNumDevicesRet = *params.pNumDevices;
  return UR_RESULT_SUCCESS;
}

// Check that the device is verified to be either a member of the context or a
// descendant of its member.
TEST(QueueDeviceCheck, CheckDeviceRestriction) {
  unittest::ScopedEnvVar EnableDefaultContexts(
      EnableDefaultContextsName, "1",
      detail::SYCLConfig<detail::SYCL_ENABLE_DEFAULT_CONTEXTS>::reset);

  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();

  UrPlatform = detail::getSyclObjImpl(Plt)->getHandleRef();
  context DefaultCtx = Plt.ext_oneapi_get_default_context();
  device Dev = DefaultCtx.get_devices()[0];

  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoAfter);
  mock::getCallbacks().set_after_callback("urDevicePartition",
                                          &redefinedDevicePartitionAfter);

  // Device is a member of the context.
  {
    queue Q{Dev};
    EXPECT_EQ(Q.get_context().get_platform(), Plt);
    EXPECT_EQ(Q.get_context(), DefaultCtx);
    queue Q2{DefaultCtx, Dev};
  }
  // Device is a descendant of a member of the context.
  {
    ParentDevice = detail::getSyclObjImpl(Dev)->getHandleRef();
    std::vector<device> Subdevices =
        Dev.create_sub_devices<info::partition_property::partition_equally>(2);
    queue Q{Subdevices[0]};
    // OpenCL backend does not support using a descendant here yet.
    EXPECT_EQ(Q.get_context() == DefaultCtx,
              Q.get_backend() != backend::opencl);
    try {
      queue Q2{DefaultCtx, Subdevices[0]};
      EXPECT_NE(Q.get_backend(), backend::opencl);
    } catch (sycl::exception &e) {
      EXPECT_TRUE(e.code() == errc::invalid);
      EXPECT_EQ(Q.get_backend(), backend::opencl);
      EXPECT_EQ(
          std::strcmp(e.what(),
                      "Queue cannot be constructed with the given context and "
                      "device since the device is not a member of the context "
                      "(descendants of devices from the context are not "
                      "supported on OpenCL yet)."),
          0);
    }
  }
  // Device is neither of the two.
  {
    ParentDevice = nullptr;
    device Device = detail::createSyclObjFromImpl<device>(
        std::make_shared<detail::device_impl>(reinterpret_cast<ur_device_handle_t>(0x01),
                                              detail::getSyclObjImpl(Plt)));
    queue Q{Device};
    EXPECT_NE(Q.get_context(), DefaultCtx);
    try {
      queue Q2{DefaultCtx, Device};
      EXPECT_TRUE(false);
    } catch (sycl::exception &e) {
      EXPECT_TRUE(e.code() == errc::invalid);
      EXPECT_NE(
          std::strstr(e.what(),
                      "Queue cannot be constructed with the given context and "
                      "device"),
          nullptr);
    }
  }
}
} // anonymous namespace
