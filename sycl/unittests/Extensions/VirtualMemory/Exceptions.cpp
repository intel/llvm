#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>

#include "ur_api.h"

#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <array>

namespace syclext = sycl::ext::oneapi::experimental;

constexpr size_t NumberOfDevices = 3;

std::array<ur_device_handle_t, NumberOfDevices> GlobalDevicesHandle{
    mock::createDummyHandle<ur_device_handle_t>(),
    mock::createDummyHandle<ur_device_handle_t>(),
    mock::createDummyHandle<ur_device_handle_t>(),
};

ur_result_t setup_urDeviceGet(void *pParams) {
  auto params = *static_cast<ur_device_get_params_t *>(pParams);
  if (*params.ppNumDevices)
    **params.ppNumDevices = 3;
  if (*params.pphDevices) {
    if (*params.pNumEntries > 0)
      (*params.pphDevices)[0] = GlobalDevicesHandle[0];
    if (*params.pNumEntries > 1)
      (*params.pphDevices)[1] = GlobalDevicesHandle[1];
    if (*params.pNumEntries > 2)
      (*params.pphDevices)[2] = GlobalDevicesHandle[2];
  }
  return UR_RESULT_SUCCESS;
}

template <bool VirtualMemSupported>
ur_result_t after_urDeviceGetInfo_AllDevices(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT: {
    if (*params.ppPropValue)
      *static_cast<ur_bool_t *>(*params.ppPropValue) = VirtualMemSupported;
    return UR_RESULT_SUCCESS;
  }
  default:;
  }
  return UR_RESULT_SUCCESS;
}

template <bool VirtualMemSupported>
ur_result_t after_urDeviceGetInfo_SingleDevice(void *pParams) {
  auto params = *static_cast<ur_device_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT: {
    if (*params.ppPropValue) {
      if (*params.phDevice == GlobalDevicesHandle[0]) {
        *static_cast<ur_bool_t *>(*params.ppPropValue) = VirtualMemSupported;
      }
    }
    return UR_RESULT_SUCCESS;
  }
  default:;
  }
  return UR_RESULT_SUCCESS;
}

TEST(VirtualMemoryMultipleDevices, ThrowExceptionForGetMemGranularityContext) {

  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urDeviceGet", &setup_urDeviceGet);
  mock::getCallbacks().set_after_callback(
      "urDeviceGetInfo", &after_urDeviceGetInfo_SingleDevice<false>);
  sycl::platform Platform = sycl::platform();
  sycl::context Context{Platform};

  try {
    syclext::get_mem_granularity(Context,
                                 syclext::granularity_mode::recommended);
    FAIL() << "No exception thrown.";
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::feature_not_supported);
    EXPECT_STREQ(e.what(), "One or more devices in the context does not "
                           "support aspect::ext_oneapi_virtual_mem.");
  }

  try {
    syclext::get_mem_granularity(Context, syclext::granularity_mode::minimum);
    FAIL() << "No exception thrown.";
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::feature_not_supported);
    EXPECT_STREQ(e.what(), "One or more devices in the context does not "
                           "support aspect::ext_oneapi_virtual_mem.");
  }
}

TEST(VirtualMemoryMultipleDevices, ThrowExceptionForGetMemGranularityDevice) {

  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urDeviceGet", &setup_urDeviceGet);
  mock::getCallbacks().set_after_callback(
      "urDeviceGetInfo", &after_urDeviceGetInfo_AllDevices<false>);

  sycl::platform Platform = sycl::platform();
  sycl::context Context{Platform};

  try {
    syclext::get_mem_granularity(Context.get_devices()[0], Context,
                                 syclext::granularity_mode::recommended);
    FAIL() << "No exception thrown.";
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::feature_not_supported);
    EXPECT_STREQ(e.what(),
                 "Device does not support aspect::ext_oneapi_virtual_mem.");
  }

  try {
    syclext::get_mem_granularity(Context.get_devices()[0], Context,
                                 syclext::granularity_mode::minimum);
    FAIL() << "No exception thrown.";
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::feature_not_supported);
    EXPECT_STREQ(e.what(),
                 "Device does not support aspect::ext_oneapi_virtual_mem.");
  }
}

TEST(VirtualMemoryMultipleDevices, ReserveVirtualMemoryRange) {

  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urDeviceGet", &setup_urDeviceGet);
  mock::getCallbacks().set_after_callback(
      "urDeviceGetInfo", &after_urDeviceGetInfo_SingleDevice<false>);
  sycl::platform Platform = sycl::platform();
  sycl::context Context{Platform};

  try {
    syclext::reserve_virtual_mem(0, sizeof(int), Context);
    FAIL() << "No exception thrown.";
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::feature_not_supported);
    EXPECT_STREQ(e.what(),
                 "One or more devices in the supplied context does not support "
                 "aspect::ext_oneapi_virtual_mem.");
  }
}

TEST(VirtualMemoryMultipleDevices, ReservePhysicalMemory) {

  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_after_callback("urDeviceGet", &setup_urDeviceGet);
  mock::getCallbacks().set_after_callback(
      "urDeviceGetInfo", &after_urDeviceGetInfo_AllDevices<false>);
  sycl::platform Platform = sycl::platform();
  sycl::context Context{Platform};

  try {
    syclext::physical_mem PhysicalMem{Context.get_devices()[0], Context,
                                      sizeof(int)};
    FAIL() << "No exception thrown.";
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::feature_not_supported);
    EXPECT_STREQ(e.what(),
                 "Device does not support aspect::ext_oneapi_virtual_mem.");
  }
}
