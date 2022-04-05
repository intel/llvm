//==------- aspect-selector.cpp --- Check aspect device selector -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/device_selector.hpp>

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

#include <vector>

using namespace sycl;

std::vector<aspect> Required, Denied;
std::vector<std::vector<aspect>> Available;

pi_result redefinedDeviceRetain(pi_device) { return PI_SUCCESS; }
pi_result redefinedDeviceRelease(pi_device) { return PI_SUCCESS; }

platform *Platform;

pi_result redefinedDeviceGetInfo(pi_device device, pi_device_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t *param_value_size_ret) {
  long DevIdx = reinterpret_cast<long>(device) - 1;

  assert(DevIdx >= 0 && (unsigned)(DevIdx) < Available.size());

  const std::vector<aspect> &Aspects = Available[DevIdx];

  if (param_name == PI_DEVICE_INFO_TYPE) {
    static const std::map<aspect, pi_device_type> Types = {
        {aspect::cpu, PI_DEVICE_TYPE_CPU},
        {aspect::gpu, PI_DEVICE_TYPE_GPU},
        {aspect::accelerator, PI_DEVICE_TYPE_ACC}};

    if (param_value) {
      bool DeviceTypeSet = false;
      for (const auto &P : Types) {
        auto It = std::find(Aspects.begin(), Aspects.end(), P.first);
        if (It != Aspects.end()) {
          *reinterpret_cast<pi_device_type *>(param_value) = P.second;
          DeviceTypeSet = true;
          break;
        }
      }

      assert(DeviceTypeSet);
    }

    return PI_SUCCESS;
  }

  if (param_name == PI_DEVICE_INFO_PARENT_DEVICE) {
    if (param_value)
      *reinterpret_cast<pi_device *>(param_value) = NULL;

    return PI_SUCCESS;
  }

  if (param_name == PI_DEVICE_INFO_EXTENSIONS) {
    if (param_value_size_ret)
      *param_value_size_ret = 0;
    return PI_SUCCESS;
  }

  if (param_name == PI_DEVICE_INFO_PLATFORM) {
    if (param_value)
      *reinterpret_cast<pi_platform *>(param_value) =
          detail::getSyclObjImpl(*Platform)->getHandleRef();
    return PI_SUCCESS;
  }

  if (param_name == PI_DEVICE_INFO_NAME) {
    const std::string DevName = "Test Device # " + std::to_string(DevIdx);
    if (param_value_size_ret)
      *param_value_size_ret = DevName.length();
    if (param_value) {
      size_t L = std::min(DevName.length(), param_value_size);
      memcpy(param_value, DevName.data(), L);
    }

    return PI_SUCCESS;
  }

  return PI_INVALID_VALUE;
}

pi_result redefinedDevicesGet(pi_platform Plt, pi_device_type DeviceType,
                              pi_uint32 NumEntries, pi_device *Devs,
                              pi_uint32 *NumDevices) {
  if (NumDevices)
    *NumDevices = Available.size();

  if (Devs) {
    long MaxIdx = std::min((size_t)NumEntries, Available.size());

    for (int DevIdx = 0; DevIdx < MaxIdx; ++DevIdx)
      Devs[DevIdx] = reinterpret_cast<pi_device>(DevIdx + 1);
  }

  return PI_SUCCESS;
}

class AspectSelector : public ::testing::Test {
public:
  AspectSelector() = default;

protected:
  static void SetUpTestSuite() {
    MPlt = new platform{sycl::default_selector()};
    if (MPlt->is_host()) {
      return;
    }

    Platform = MPlt;

    MMock = new unittest::PiMock(*MPlt);

    using namespace sycl::detail;

    setupDefaultMockAPIs(*MMock);
    MMock->redefine<PiApiKind::piDeviceGetInfo>(redefinedDeviceGetInfo);
    MMock->redefine<PiApiKind::piDevicesGet>(redefinedDevicesGet);
    MMock->redefine<PiApiKind::piDeviceRelease>(redefinedDeviceRelease);
    MMock->redefine<PiApiKind::piDeviceRetain>(redefinedDeviceRetain);
  }

  static void TearDownTestSuite() {
    Platform = nullptr;
    delete MMock;
    delete MPlt;
  }

  static platform *MPlt;
  static unittest::PiMock *MMock;
};

platform *AspectSelector::MPlt;
unittest::PiMock *AspectSelector::MMock;

TEST_F(AspectSelector, TestPositive) {
  if (MPlt->is_host()) {
    GTEST_SKIP() << "This test is not supported for host\n";
    return;
  }

  {
    Required = {aspect::cpu};
    Denied.clear();
    Available = {{aspect::cpu}, {aspect::gpu}, {aspect::accelerator}};
    auto Selector = aspect_selector(Required, Denied);
  }
}

TEST_F(AspectSelector, TestNegative) {
  if (MPlt->is_host()) {
    GTEST_SKIP() << "This test is not supported for host\n";
    return;
  }

  {
    Required = {aspect::cpu};
    Denied.clear();
    Available = {{aspect::cpu}, {aspect::gpu}, {aspect::accelerator}};

    auto Selector = aspect_selector<aspect::cpu, aspect::gpu>();

    try {
      device D = Selector.select_device();
      ASSERT_TRUE(false && "Unexpected device returned for both CPU and GPU");
    } catch (const sycl::exception &E) {
      ASSERT_EQ(E.code().value(), static_cast<int>(errc::runtime));
      const std::string_view Msg{E.what()};
      const std::string MsgTpl = "No device of requested type available";
      ASSERT_TRUE(Msg.find(MsgTpl) != Msg.npos);
      std::cout << "Exception: " << E.what() << std::endl;
    }
  }
}
