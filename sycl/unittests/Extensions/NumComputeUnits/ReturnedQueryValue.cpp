#include <sycl/sycl.hpp>

#include "ur_api.h"

#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>

template <uint32_t ExpectedValue>
ur_result_t after_urDeviceGetInfo(void *pParams) {
  auto params = reinterpret_cast<ur_device_get_info_params_t *>(pParams);
  if (*params->ppropName == UR_DEVICE_INFO_NUM_COMPUTE_UNITS) {
    if (*params->ppPropValue)
      *static_cast<uint32_t *>(*params->ppPropValue) = ExpectedValue;
    if (*params->ppPropSizeRet)
      **params->ppPropSizeRet = sizeof(uint32_t);
  }
  return UR_RESULT_SUCCESS;
}

TEST(NumComputeUnitsTests, CheckExpectedValue) {

  constexpr uint32_t ExpectedNumComputeUnits = 111;

  sycl::unittest::UrMock<> Mock;
  sycl::platform Platform = sycl::platform();
  sycl::queue Queue{Platform.get_devices()[0]};

  mock::getCallbacks().set_after_callback(
      "urDeviceGetInfo", &after_urDeviceGetInfo<ExpectedNumComputeUnits>);

  size_t NumberComputeUnits =
      Queue.get_device()
          .get_info<sycl::ext::oneapi::info::device::num_compute_units>();

  EXPECT_EQ(NumberComputeUnits, ExpectedNumComputeUnits);
}
