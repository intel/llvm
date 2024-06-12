#include <sycl/sycl.hpp>

#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <algorithm>

namespace {
const auto COMPOSITE_DEVICE = reinterpret_cast<ur_device_handle_t>(1u);
const auto COMPONENT_DEVICE_A = reinterpret_cast<ur_device_handle_t>(2u);
const auto COMPONENT_DEVICE_B = reinterpret_cast<ur_device_handle_t>(3u);

ur_result_t redefine_urDeviceGet(void *pParams) {
  auto params = *static_cast<ur_device_get_params_t *>(pParams);
  if (*params.ppNumDevices)
    **params.ppNumDevices = 2;
  if (*params.pphDevices) {
    if (*params.pNumEntries > 0)
      (*params.pphDevices)[0] = COMPONENT_DEVICE_A;
    if (*params.pNumEntries > 1)
      (*params.pphDevices)[1] = COMPONENT_DEVICE_B;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t after_urDeviceGetInfo(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_DEVICE_INFO_COMPOSITE_DEVICE:
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(ur_device_handle_t);
    if (*params.ppPropValue) {
      if (*params.phDevice == COMPONENT_DEVICE_A ||
          *params.phDevice == COMPONENT_DEVICE_B) {
        *static_cast<ur_device_handle_t *>(*params.ppPropValue) =
            COMPOSITE_DEVICE;
      } else
        *static_cast<ur_device_handle_t *>(*params.ppPropValue) = nullptr;
    }

    return UR_RESULT_SUCCESS;

  case UR_DEVICE_INFO_COMPONENT_DEVICES:
    if (*params.phDevice == COMPOSITE_DEVICE) {
      if (*params.ppPropSizeRet)
        **params.ppPropSizeRet = 2 * sizeof(ur_device_handle_t);
      if (*params.ppPropValue) {
        if (*params.ppropSize >= sizeof(ur_device_handle_t))
          static_cast<ur_device_handle_t *>(*params.ppPropValue)[0] =
              COMPONENT_DEVICE_A;
        if (*params.ppropSize >= 2 * sizeof(ur_device_handle_t))
          static_cast<ur_device_handle_t *>(*params.ppPropValue)[1] =
              COMPONENT_DEVICE_B;
      }

    } else {
      if (*params.ppPropSizeRet)
        **params.ppPropSizeRet = 0;
    }

    return UR_RESULT_SUCCESS;

  default:
    return UR_RESULT_SUCCESS;
  }
}

ur_result_t after_urDeviceGetInfo_unsupported(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_DEVICE_INFO_COMPOSITE_DEVICE:
  case UR_DEVICE_INFO_COMPONENT_DEVICES:
    return UR_RESULT_ERROR_INVALID_VALUE;

  default:
    return UR_RESULT_SUCCESS;
  }
}

ur_result_t after_urDeviceGetInfo_no_component_devices(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_DEVICE_INFO_COMPOSITE_DEVICE:
    return UR_RESULT_ERROR_INVALID_VALUE;
  case UR_DEVICE_INFO_COMPONENT_DEVICES:
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = 0;
    return UR_RESULT_SUCCESS;

  default:
    return UR_RESULT_SUCCESS;
  }
}

thread_local std::vector<ur_device_handle_t> DevicesUsedInContextCreation;

ur_result_t after_urContextCreate(void *pParams) {
  auto params = *static_cast<ur_context_create_params_t *>(pParams);
  DevicesUsedInContextCreation.assign(
      *params.pphDevices, *params.pphDevices + *params.pDeviceCount);

  return UR_RESULT_SUCCESS;
}

} // namespace

TEST(CompositeDeviceTest, DescendentDeviceSupportInContext) {
  sycl::unittest::UrMock<sycl::backend::ext_oneapi_level_zero> Mock;
  mock::getCallbacks().set_replace_callback("urDeviceGet",
                                            &redefine_urDeviceGet);
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo);
  mock::getCallbacks().set_after_callback("urContextCreate",
                                          &after_urContextCreate);

  sycl::platform Plt = sycl::platform();
  ASSERT_EQ(Plt.get_backend(), sycl::backend::ext_oneapi_level_zero);

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
  ASSERT_TRUE(std::any_of(DevicesUsedInContextCreation.begin(),
                          DevicesUsedInContextCreation.end(),
                          [=](ur_device_handle_t D) { return D == COMPOSITE_DEVICE; }));
  ASSERT_TRUE(std::any_of(
      DevicesUsedInContextCreation.begin(), DevicesUsedInContextCreation.end(),
      [=](ur_device_handle_t D) { return D == COMPONENT_DEVICE_A; }));
  ASSERT_TRUE(std::any_of(
      DevicesUsedInContextCreation.begin(), DevicesUsedInContextCreation.end(),
      [=](ur_device_handle_t D) { return D == COMPONENT_DEVICE_B; }));
  // Even though under the hood we have created context for 3 devices,
  // user-visible interface should only report the exact list of devices passed
  // by user to the context constructor.
  ASSERT_EQ(CompositeDevContext.get_devices().size(), 1u);
  ASSERT_EQ(CompositeDevContext.get_devices().front(), CompositeDevice);
}

TEST(CompositeDeviceTest, DescendentDeviceSupportInQueue) {
  sycl::unittest::UrMock<sycl::backend::ext_oneapi_level_zero> Mock;
  mock::getCallbacks().set_replace_callback("urDeviceGet",
                                            &redefine_urDeviceGet);
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo);
  mock::getCallbacks().set_after_callback("urContextCreate",
                                          &after_urContextCreate);

  sycl::platform Plt = sycl::platform();
  ASSERT_EQ(Plt.get_backend(), sycl::backend::ext_oneapi_level_zero);

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
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urDeviceGet",
                                            &redefine_urDeviceGet);
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &after_urDeviceGetInfo_unsupported);

  sycl::platform Plt = sycl::platform();

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
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urDeviceGet",
                                            &redefine_urDeviceGet);
  mock::getCallbacks().set_after_callback(
      "urDeviceGetInfo", &after_urDeviceGetInfo_no_component_devices);

  sycl::platform Plt = sycl::platform();

  sycl::device ComponentDevice = Plt.get_devices()[0];
  ASSERT_FALSE(ComponentDevice.has(sycl::aspect::ext_oneapi_is_composite));

  std::vector<sycl::device> ComponentDevices = ComponentDevice.get_info<
      sycl::ext::oneapi::experimental::info::device::component_devices>();
  ASSERT_TRUE(ComponentDevices.empty());
}
