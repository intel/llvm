#include <sycl/sycl.hpp>

#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

#include <algorithm>

namespace {
const auto COMPOSITE_DEVICE_0 = reinterpret_cast<pi_device>(1u);
const auto COMPONENT_DEVICE_A = reinterpret_cast<pi_device>(2u);
const auto COMPONENT_DEVICE_B = reinterpret_cast<pi_device>(3u);

// We do not report COMPONENT_DEVICE_D through mocked piDevicesGet to emulate
// that it is not available to ensure that COMPOSITE_DEVICE_1 is not returned
// through platform::ext_oneapi_get_composite_devices and
// sycl:ext::oneapi::experimental::get_composite_devices APIs
const auto COMPOSITE_DEVICE_1 = reinterpret_cast<pi_device>(4u);
const auto COMPONENT_DEVICE_C = reinterpret_cast<pi_device>(5u);
const auto COMPONENT_DEVICE_D = reinterpret_cast<pi_device>(6u);

pi_result redefine_piDevicesGet(pi_platform platform, pi_device_type,
                                pi_uint32 num_entries, pi_device *devices,
                                pi_uint32 *num_devices) {
  if (num_devices)
    *num_devices = 3;
  if (devices) {
    if (num_entries > 0)
      devices[0] = COMPONENT_DEVICE_A;
    if (num_entries > 1)
      devices[1] = COMPONENT_DEVICE_B;
    if (num_entries > 2)
      devices[2] = COMPONENT_DEVICE_C;
  }
  return PI_SUCCESS;
}

pi_result after_piDeviceGetInfo(pi_device device, pi_device_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_EXT_ONEAPI_DEVICE_INFO_COMPOSITE_DEVICE:
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(pi_device);
    if (param_value) {
      if (device == COMPONENT_DEVICE_A || device == COMPONENT_DEVICE_B) {
        *static_cast<pi_device *>(param_value) = COMPOSITE_DEVICE_0;
      } else if (device == COMPONENT_DEVICE_C || device == COMPONENT_DEVICE_D) {
        *static_cast<pi_device *>(param_value) = COMPOSITE_DEVICE_1;
      } else
        *static_cast<pi_device *>(param_value) = nullptr;
    }

    return PI_SUCCESS;

  case PI_EXT_ONEAPI_DEVICE_INFO_COMPONENT_DEVICES:
    if (device == COMPOSITE_DEVICE_0) {
      if (param_value_size_ret)
        *param_value_size_ret = 2 * sizeof(pi_device);
      if (param_value) {
        if (param_value_size >= sizeof(pi_device))
          static_cast<pi_device *>(param_value)[0] = COMPONENT_DEVICE_A;
        if (param_value_size >= 2 * sizeof(pi_device))
          static_cast<pi_device *>(param_value)[1] = COMPONENT_DEVICE_B;
      }
    } else if (device == COMPOSITE_DEVICE_1) {
      if (param_value_size_ret)
        *param_value_size_ret = 2 * sizeof(pi_device);
      if (param_value) {
        if (param_value_size >= sizeof(pi_device))
          static_cast<pi_device *>(param_value)[0] = COMPONENT_DEVICE_C;
        if (param_value_size >= 2 * sizeof(pi_device))
          static_cast<pi_device *>(param_value)[1] = COMPONENT_DEVICE_D;
      }
    } else {
      if (param_value_size_ret)
        *param_value_size_ret = 0;
    }

    return PI_SUCCESS;

  default:
    return PI_SUCCESS;
  }
}

pi_result after_piDeviceGetInfo_unsupported(pi_device device,
                                            pi_device_info param_name,
                                            size_t param_value_size,
                                            void *param_value,
                                            size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_EXT_ONEAPI_DEVICE_INFO_COMPOSITE_DEVICE:
  case PI_EXT_ONEAPI_DEVICE_INFO_COMPONENT_DEVICES:
    return PI_ERROR_INVALID_VALUE;

  default:
    return PI_SUCCESS;
  }
}

pi_result after_piDeviceGetInfo_no_component_devices(
    pi_device device, pi_device_info param_name, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_EXT_ONEAPI_DEVICE_INFO_COMPOSITE_DEVICE:
    return PI_ERROR_INVALID_VALUE;
  case PI_EXT_ONEAPI_DEVICE_INFO_COMPONENT_DEVICES:
    if (param_value_size_ret)
      *param_value_size_ret = 0;
    return PI_SUCCESS;

  default:
    return PI_SUCCESS;
  }
}

thread_local std::vector<pi_device> DevicesUsedInContextCreation;

pi_result after_piContextCreate(const pi_context_properties *,
                                pi_uint32 num_devices, const pi_device *devices,
                                void (*)(const char *, const void *, size_t,
                                         void *),
                                void *, pi_context *ret_context) {

  DevicesUsedInContextCreation.assign(devices, devices + num_devices);

  return PI_SUCCESS;
}

} // namespace

TEST(CompositeDeviceTest, PlatformExtOneAPIGetCompositeDevices) {
  sycl::unittest::PiMock Mock;
  Mock.redefine<sycl::detail::PiApiKind::piDevicesGet>(redefine_piDevicesGet);
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      after_piDeviceGetInfo);

  sycl::platform Plt = Mock.getPlatform();

  std::vector<sycl::device> Composites = Plt.ext_oneapi_get_composite_devices();
  // We don't expect to see COMPOSITE_DEVICE_1 here, because one of its
  // components (COMPONENT_DEVICE_D) is not available.
  ASSERT_EQ(Composites.size(), 1u);
  ASSERT_EQ(sycl::bit_cast<pi_device>(
                sycl::get_native<sycl::backend::opencl>(Composites.front())),
            COMPOSITE_DEVICE_0);
}

TEST(CompositeDeviceTest, SYCLExtOneAPIExperimentalGetCompositeDevices) {
  sycl::unittest::PiMock Mock;
  Mock.redefine<sycl::detail::PiApiKind::piDevicesGet>(redefine_piDevicesGet);
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      after_piDeviceGetInfo);

  sycl::platform Plt = Mock.getPlatform();

  std::vector<sycl::device> Composites =
      sycl::ext::oneapi::experimental::get_composite_devices();
  // We don't expect to see COMPOSITE_DEVICE_1 here, because one of its
  // components (COMPONENT_DEVICE_D) is not available.
  ASSERT_EQ(Composites.size(), 1u);
  ASSERT_EQ(sycl::bit_cast<pi_device>(
                sycl::get_native<sycl::backend::opencl>(Composites.front())),
            COMPOSITE_DEVICE_0);
}

TEST(CompositeDeviceTest, DescendentDeviceSupportInContext) {
  sycl::unittest::PiMock Mock;
  Mock.redefine<sycl::detail::PiApiKind::piDevicesGet>(redefine_piDevicesGet);
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      after_piDeviceGetInfo);
  Mock.redefineAfter<sycl::detail::PiApiKind::piContextCreate>(
      after_piContextCreate);

  sycl::platform Plt = Mock.getPlatform();
  sycl::device RootDevice = Plt.get_devices()[0];
  ASSERT_TRUE(RootDevice.has(sycl::aspect::ext_oneapi_is_component));
  sycl::context Ctx(RootDevice);
  // We expect to only see the passed device
  ASSERT_EQ(DevicesUsedInContextCreation.size(), 1u);
  ASSERT_EQ(DevicesUsedInContextCreation.front(), COMPONENT_DEVICE_A);

  auto CompositeDevice = RootDevice.get_info<
      sycl::ext::oneapi::experimental::info::device::composite_device>();
  sycl::context CompositeDevContext(CompositeDevice);
  // To make sure that component devices can also be used within a context
  // created for a composite device, we expect them to be implicitly added to
  // the context under the hood.
  ASSERT_EQ(DevicesUsedInContextCreation.size(), 3u);
  ASSERT_TRUE(std::any_of(
      DevicesUsedInContextCreation.begin(), DevicesUsedInContextCreation.end(),
      [=](pi_device D) { return D == COMPOSITE_DEVICE_0; }));
  ASSERT_TRUE(std::any_of(
      DevicesUsedInContextCreation.begin(), DevicesUsedInContextCreation.end(),
      [=](pi_device D) { return D == COMPONENT_DEVICE_A; }));
  ASSERT_TRUE(std::any_of(
      DevicesUsedInContextCreation.begin(), DevicesUsedInContextCreation.end(),
      [=](pi_device D) { return D == COMPONENT_DEVICE_B; }));
  // Even though under the hood we have created context for 3 devices,
  // user-visible interface should only report the exact list of devices passed
  // by user to the context constructor.
  ASSERT_EQ(CompositeDevContext.get_devices().size(), 1u);
  ASSERT_EQ(CompositeDevContext.get_devices().front(), CompositeDevice);
}

TEST(CompositeDeviceTest, DescendentDeviceSupportInQueue) {
  sycl::unittest::PiMock Mock;
  Mock.redefine<sycl::detail::PiApiKind::piDevicesGet>(redefine_piDevicesGet);
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      after_piDeviceGetInfo);
  Mock.redefineAfter<sycl::detail::PiApiKind::piContextCreate>(
      after_piContextCreate);

  sycl::platform Plt = Mock.getPlatform();
  sycl::device ComponentDevice = Plt.get_devices()[0];
  ASSERT_TRUE(ComponentDevice.has(sycl::aspect::ext_oneapi_is_component));

  auto CompositeDevice = ComponentDevice.get_info<
      sycl::ext::oneapi::experimental::info::device::composite_device>();
  sycl::context CompositeDevContext(CompositeDevice);
  // Component device should be implicitly usable as part of composite context,
  // so there should be no errors during queue creation below.
  sycl::queue Queue(CompositeDevContext, ComponentDevice);
}

TEST(CompositeDeviceTest, UnsupportedNegative) {
  // For the unsupported case, the backend does not need to be L0.
  sycl::unittest::PiMock Mock;
  Mock.redefine<sycl::detail::PiApiKind::piDevicesGet>(redefine_piDevicesGet);
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      after_piDeviceGetInfo_unsupported);

  sycl::platform Plt = Mock.getPlatform();

  sycl::device ComponentDevice = Plt.get_devices()[0];
  ASSERT_FALSE(ComponentDevice.has(sycl::aspect::ext_oneapi_is_component));

  try {
    std::ignore = ComponentDevice.get_info<
        sycl::ext::oneapi::experimental::info::device::composite_device>();
  } catch (sycl::exception &E) {
    ASSERT_EQ(E.code(), sycl::make_error_code(sycl::errc::invalid));
  }
}

TEST(CompositeDeviceTest, NoComponentDevices) {
  sycl::unittest::PiMock Mock;
  Mock.redefine<sycl::detail::PiApiKind::piDevicesGet>(redefine_piDevicesGet);
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      after_piDeviceGetInfo_no_component_devices);

  sycl::platform Plt = Mock.getPlatform();

  sycl::device ComponentDevice = Plt.get_devices()[0];
  ASSERT_FALSE(ComponentDevice.has(sycl::aspect::ext_oneapi_is_composite));

  std::vector<sycl::device> ComponentDevices = ComponentDevice.get_info<
      sycl::ext::oneapi::experimental::info::device::component_devices>();
  ASSERT_TRUE(ComponentDevices.empty());
}
