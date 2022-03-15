//==------- aspect-selector.cpp --- Check aspect device selector -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

using namespace sycl;

std::vector<aspect> Required, Denied;
std::vector<std::vector<aspect>> Available;

pi_result redefinedDeviceGetInfo(pi_device device,
                                 pi_device_info param_name,
                                 size_t param_value_size,
                                 void *param_value,
                                 size_t *param_value_size_ret) {
  int DevIdx = *reinterpret_cast<int *>(device);

  assert(DevIdx >= 0 && (unsigned)(DevIdx) < Available.size());

  const std::vector<aspect> &Aspects = Available[DevIdx];

  if (param_name == PI_DEVICE_INFO_TYPE) {
    bool DeviceTypeSet = false;
    auto It = std::find(Aspects.begin(), Aspects.end(), aspect::cpu);
    if (It != Aspects.end()) {
      DeviceTypeSet = true;
      if (param_value)
        *reinterpret_cast<pi_device_type *>(param_value) = PI_DEVICE_TYPE_CPU;
      if (param_value_size_ret)
        *param_value_size_ret = sizeof(pi_device_type);
    }

    It = std::find(Aspects.begin(), Aspects.end(), aspect::gpu);
    if (It != Aspects.end()) {
      DeviceTypeSet = true;
      if (param_value)
        *reinterpret_cast<pi_device_type *>(param_value) = PI_DEVICE_TYPE_GPU;
      if (param_value_size_ret)
        *param_value_size_ret = sizeof(pi_device_type);
    }

    It = std::find(Aspects.begin(), Aspects.end(), aspect::accelerator);
    if (It != Aspects.end()) {
      DeviceTypeSet = true;
      if (param_value)
        *reinterpret_cast<pi_device_type *>(param_value) = PI_DEVICE_TYPE_ACC;
      if (param_value_size_ret)
        *param_value_size_ret = sizeof(pi_device_type);
    }

    assert(DeviceTypeSet);
  }

  return PI_INVALID_VALUE;
}


static void setupMock(unittest::PiMock &Mock) {
  using namespace sycl::detail;

  setupDefaultMockAPIs(Mock);
  Mock.redefine<PiApiKind::piDeviceGetInfo>(redefinedDeviceGetInfo);
}

TEST(AspectSelector, TestPositive) {
  sycl::platform Plt{sycl::default_selector()};
  sycl::unittest::PiMock Mock{Plt};
  setupMock(Mock);

  {
    Required = {aspect::cpu};
    Denied.clear();
    Available = {{aspect::cpu}, {aspect::gpu}, {aspect::accelerator}};
    // TODO set available list
    auto Selector = aspect_selector(Required, Denied);
  }
}

TEST(AspectSelector, TestNegative) {
  auto Selector = aspect_selector<aspect::cpu, aspect::gpu>();

  try {
    device D = Selector.select_device();
    ASSERT_TRUE(false && "Unexpected device returned for both CPU and GPU");
  } catch (const sycl::exception &E) {
  }
}

