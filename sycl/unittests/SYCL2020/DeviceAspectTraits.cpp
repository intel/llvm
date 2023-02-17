#include <sycl/sycl.hpp>

#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

TEST(DeviceAspectTraits, AnyDeviceHasHost) {
  std::cout << sycl::any_device_has_v<sycl::aspect::host> << std::endl;
#define __SYCL_ASPECT(ASPECT, ASPECT_VAL)                                      \
  constexpr bool CheckAnyDeviceHas##ASPECT =                                   \
      sycl::any_device_has_v<sycl::aspect::ASPECT>;                            \
  constexpr bool CheckAllDevicesHave##ASPECT =                                 \
      sycl::all_devices_have_v<sycl::aspect::ASPECT>;

#include <sycl/info/aspects.def>

#undef __SYCL_ASPECT
}
