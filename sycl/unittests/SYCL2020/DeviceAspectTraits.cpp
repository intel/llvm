#include <sycl/sycl.hpp>

#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

TEST(DeviceAspectTraits, AnyDeviceHasAspect) {
#define __SYCL_ASPECT(ASPECT, ASPECT_VAL)                                      \
  constexpr bool CheckAnyDeviceHas##ASPECT =                                   \
      sycl::any_device_has_v<sycl::aspect::ASPECT>;                            \
  std::ignore = CheckAnyDeviceHas##ASPECT;

#include <sycl/info/aspects.def>

#undef __SYCL_ASPECT
}
TEST(DeviceAspectTraits, AllDevicesHaveAspect) {
#define __SYCL_ASPECT(ASPECT, ASPECT_VAL)                                      \
  constexpr bool CheckAllDevicesHave##ASPECT =                                 \
      sycl::all_devices_have_v<sycl::aspect::ASPECT>;                          \
  std::ignore = CheckAllDevicesHave##ASPECT;

#include <sycl/info/aspects.def>

#undef __SYCL_ASPECT
}
