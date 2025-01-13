//==- DeviceAspectTraits.cpp --- any_device_has/all_devices_have unit test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#include <sycl/sycl.hpp>

#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)                    \
  __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE)

TEST(DeviceAspectTraits, AnyDeviceHasAspect) {
#define __SYCL_ASPECT(ASPECT, ASPECT_VAL)                                      \
  constexpr bool CheckAnyDeviceHas##ASPECT =                                   \
      sycl::any_device_has_v<sycl::aspect::ASPECT>;                            \
  std::ignore = CheckAnyDeviceHas##ASPECT;

#include <sycl/info/aspects.def>

#undef __SYCL_ASPECT

#define __SYCL_ASPECT_DEPRECATED(ASPECT, ASPECT_VAL, MSG)                      \
  constexpr bool CheckAnyDeviceHas##ASPECT =                                   \
      sycl::any_device_has_v<sycl::aspect::ASPECT>;                            \
  std::ignore = CheckAnyDeviceHas##ASPECT;

#include <sycl/info/aspects_deprecated.def>

#undef __SYCL_ASPECT_DEPRECATED
}
TEST(DeviceAspectTraits, AllDevicesHaveAspect) {
#define __SYCL_ASPECT(ASPECT, ASPECT_VAL)                                      \
  constexpr bool CheckAllDevicesHave##ASPECT =                                 \
      sycl::all_devices_have_v<sycl::aspect::ASPECT>;                          \
  std::ignore = CheckAllDevicesHave##ASPECT;

#include <sycl/info/aspects.def>

#undef __SYCL_ASPECT

#define __SYCL_ASPECT_DEPRECATED(ASPECT, ASPECT_VAL, MSG)                      \
  constexpr bool CheckAllDevicesHave##ASPECT =                                 \
      sycl::all_devices_have_v<sycl::aspect::ASPECT>;                          \
  std::ignore = CheckAllDevicesHave##ASPECT;

#include <sycl/info/aspects_deprecated.def>

#undef __SYCL_ASPECT_DEPRECATED
}

#undef __SYCL_ASPECT_DEPRECATED_ALIAS
