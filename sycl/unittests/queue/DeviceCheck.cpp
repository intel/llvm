//==----------------- DeviceCheck.cpp --- queue unit tests -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <detail/config.hpp>
#include <detail/device_impl.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>
#include <helpers/ScopedEnvVar.hpp>

using namespace sycl;

namespace {

inline constexpr auto EnableDefaultContextsName =
    "SYCL_ENABLE_DEFAULT_CONTEXTS";

pi_result redefinedContextCreate(const pi_context_properties *properties,
                                 pi_uint32 num_devices,
                                 const pi_device *devices,
                                 void (*pfn_notify)(const char *errinfo,
                                                    const void *private_info,
                                                    size_t cb, void *user_data),
                                 void *user_data, pi_context *ret_context) {
  *ret_context = reinterpret_cast<pi_context>(1);
  return PI_SUCCESS;
}

pi_result redefinedContextRelease(pi_context context) { return PI_SUCCESS; }

pi_device ParentDevice = nullptr;
pi_platform PiPlatform = nullptr;

pi_result redefinedDeviceGetInfo(pi_device device, pi_device_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t *param_value_size_ret) {
  if (param_name == PI_DEVICE_INFO_PARTITION_PROPERTIES) {
    if (param_value) {
      auto *Result =
          reinterpret_cast<pi_device_partition_property *>(param_value);
      *Result = PI_DEVICE_PARTITION_EQUALLY;
    }
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(pi_device_partition_property);
  } else if (param_name == PI_DEVICE_INFO_MAX_COMPUTE_UNITS) {
    auto *Result = reinterpret_cast<pi_uint32 *>(param_value);
    *Result = 2;
  } else if (param_name == PI_DEVICE_INFO_PARENT_DEVICE) {
    auto *Result = reinterpret_cast<pi_device *>(param_value);
    *Result = (device == ParentDevice) ? nullptr : ParentDevice;
  } else if (param_name == PI_DEVICE_INFO_PLATFORM) {
    auto *Result = reinterpret_cast<pi_platform *>(param_value);
    *Result = PiPlatform;
  } else if (param_name == PI_DEVICE_INFO_EXTENSIONS) {
    if (param_value_size_ret) {
      *param_value_size_ret = 0;
    }
  }
  return PI_SUCCESS;
}

pi_result redefinedDevicePartition(
    pi_device device, const pi_device_partition_property *properties,
    pi_uint32 num_devices, pi_device *out_devices, pi_uint32 *out_num_devices) {
  if (out_devices) {
    for (pi_uint32 I = 0; I < num_devices; ++I) {
      out_devices[I] = reinterpret_cast<pi_device>(1);
    }
  }
  if (out_num_devices)
    *out_num_devices = num_devices;
  return PI_SUCCESS;
}

pi_result redefinedDeviceRetain(pi_device device) { return PI_SUCCESS; }

pi_result redefinedDeviceRelease(pi_device device) { return PI_SUCCESS; }

pi_result redefinedQueueCreate(pi_context context, pi_device device,
                               pi_queue_properties properties,
                               pi_queue *queue) {
  return PI_SUCCESS;
}

pi_result redefinedQueueRelease(pi_queue queue) { return PI_SUCCESS; }

// Check that the device is verified to be either a member of the context or a
// descendant of its member.
TEST(QueueDeviceCheck, CheckDeviceRestriction) {
  unittest::ScopedEnvVar EnableDefaultContexts(
      EnableDefaultContextsName, "1",
      detail::SYCLConfig<detail::SYCL_ENABLE_DEFAULT_CONTEXTS>::reset);

  platform Plt{default_selector()};
  if (Plt.is_host()) {
    std::cout << "The test is not supported on host, skipping" << std::endl;
    GTEST_SKIP();
  }
  PiPlatform = detail::getSyclObjImpl(Plt)->getHandleRef();
  // Create default context normally to avoid issues during its release, which
  // takes plase after Mock is destroyed.
  context DefaultCtx = Plt.ext_oneapi_get_default_context();
  device Dev = DefaultCtx.get_devices()[0];

  unittest::PiMock Mock{Plt};
  Mock.redefine<detail::PiApiKind::piContextCreate>(redefinedContextCreate);
  Mock.redefine<detail::PiApiKind::piContextRelease>(redefinedContextRelease);
  Mock.redefine<detail::PiApiKind::piDeviceGetInfo>(redefinedDeviceGetInfo);
  Mock.redefine<detail::PiApiKind::piDevicePartition>(redefinedDevicePartition);
  Mock.redefine<detail::PiApiKind::piDeviceRelease>(redefinedDeviceRelease);
  Mock.redefine<detail::PiApiKind::piDeviceRetain>(redefinedDeviceRetain);
  Mock.redefine<detail::PiApiKind::piQueueCreate>(redefinedQueueCreate);
  Mock.redefine<detail::PiApiKind::piQueueRelease>(redefinedQueueRelease);

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
    } catch (sycl::invalid_object_error &e) {
      EXPECT_EQ(Q.get_backend(), backend::opencl);
      EXPECT_EQ(std::strcmp(
                    e.what(),
                    "Queue cannot be constructed with the given context and "
                    "device since the device is not a member of the context "
                    "(descendants of devices from the context are not "
                    "supported on OpenCL yet). -33 (PI_ERROR_INVALID_DEVICE)"),
                0);
    }
  }
  // Device is neither of the two.
  {
    ParentDevice = nullptr;
    device Device = detail::createSyclObjFromImpl<device>(
        std::make_shared<detail::device_impl>(reinterpret_cast<pi_device>(0x01),
                                              detail::getSyclObjImpl(Plt)));
    queue Q{Device};
    EXPECT_NE(Q.get_context(), DefaultCtx);
    try {
      queue Q2{DefaultCtx, Device};
      EXPECT_TRUE(false);
    } catch (sycl::invalid_object_error &e) {
      EXPECT_NE(
          std::strstr(e.what(),
                      "Queue cannot be constructed with the given context and "
                      "device"),
          nullptr);
    }
  }
}
} // anonymous namespace
