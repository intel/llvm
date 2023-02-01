//==---------------------- DeviceGlobal.cpp --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/pi.hpp>
#include <sycl/sycl.hpp>

#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <optional>

class DeviceGlobalTestKernel;
constexpr const char *DeviceGlobalTestKernelName = "DeviceGlobalTestKernel";
constexpr const char *DeviceGlobalName = "DeviceGlobalName";

sycl::ext::oneapi::experimental::device_global<int> DeviceGlobal;

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
template <> struct KernelInfo<DeviceGlobalTestKernel> {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return DeviceGlobalTestKernelName; }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
  static constexpr int64_t getKernelSize() { return 1; }
};
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

static sycl::unittest::PiImage generateDeviceGlobalImage() {
  using namespace sycl::unittest;

  // Call device global map initializer explicitly to mimic the integration
  // header.
  sycl::detail::device_global_map::add(&DeviceGlobal, DeviceGlobalName);

  // Insert remaining device global info into the binary.
  PiPropertySet PropSet;
  PiProperty DevGlobInfo =
      makeDeviceGlobalInfo(DeviceGlobalName, sizeof(int), 0);
  PropSet.insert(__SYCL_PI_PROPERTY_SET_SYCL_DEVICE_GLOBALS,
                 PiArray<PiProperty>{std::move(DevGlobInfo)});

  std::vector<unsigned char> Bin{10, 11, 12, 13, 14, 15}; // Random data

  PiArray<PiOffloadEntry> Entries =
      makeEmptyKernels({DeviceGlobalTestKernelName});

  PiImage Img{PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
              "",                                     // Compile options
              "",                                     // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}

namespace {
sycl::unittest::PiImage Imgs[] = {generateDeviceGlobalImage()};
sycl::unittest::PiImageArray<1> ImgArray{Imgs};

// Trackers.
thread_local std::optional<pi_event> DeviceGlobalFillEvent = std::nullopt;
thread_local std::optional<pi_event> DeviceGlobalWriteEvent = std::nullopt;
thread_local unsigned KernelCallCounter = 0;

// Markers.
thread_local bool TreatDeviceGlobalFillEventAsCompleted = false;
thread_local bool TreatDeviceGlobalWriteEventAsCompleted = false;

static pi_result after_piextUSMEnqueueMemset(pi_queue, void *, pi_int32, size_t,
                                             pi_uint32, const pi_event *,
                                             pi_event *event) {
  EXPECT_FALSE(DeviceGlobalFillEvent.has_value())
      << "piextUSMEnqueueMemset is called multiple times!";
  DeviceGlobalFillEvent = *event;
  return PI_SUCCESS;
}

pi_result after_piextEnqueueDeviceGlobalVariableWrite(
    pi_queue, pi_program, const char *, pi_bool, size_t, size_t, const void *,
    pi_uint32, const pi_event *, pi_event *event) {
  EXPECT_FALSE(DeviceGlobalWriteEvent.has_value())
      << "piextEnqueueDeviceGlobalVariableWrite is called multiple times!";
  DeviceGlobalWriteEvent = *event;
  return PI_SUCCESS;
}

pi_result after_piEventGetInfo(pi_event event, pi_event_info param_name, size_t,
                               void *param_value, size_t *) {
  if (param_name == PI_EVENT_INFO_COMMAND_EXECUTION_STATUS &&
      param_value != nullptr) {
    if ((TreatDeviceGlobalFillEventAsCompleted &&
         DeviceGlobalFillEvent.has_value() &&
         event == *DeviceGlobalFillEvent) ||
        (TreatDeviceGlobalWriteEventAsCompleted &&
         DeviceGlobalWriteEvent.has_value() &&
         event == *DeviceGlobalWriteEvent))
      *static_cast<pi_event_status *>(param_value) = PI_EVENT_COMPLETE;
    else
      *static_cast<pi_event_status *>(param_value) = PI_EVENT_SUBMITTED;
  }
  return PI_SUCCESS;
}

pi_result after_piEnqueueKernelLaunch(pi_queue, pi_kernel, pi_uint32,
                                      const size_t *, const size_t *,
                                      const size_t *,
                                      pi_uint32 num_events_in_wait_list,
                                      const pi_event *event_wait_list,
                                      pi_event *) {
  ++KernelCallCounter;
  EXPECT_TRUE(DeviceGlobalFillEvent.has_value())
      << "DeviceGlobalFillEvent has not been set. Kernel call "
      << KernelCallCounter;
  EXPECT_TRUE(DeviceGlobalWriteEvent.has_value())
      << "DeviceGlobalWriteEvent has not been set. Kernel call "
      << KernelCallCounter;

  const pi_event *EventListEnd = event_wait_list + num_events_in_wait_list;

  bool DeviceGlobalFillEventFound =
      std::find(event_wait_list, EventListEnd, *DeviceGlobalFillEvent) !=
      EventListEnd;
  if (TreatDeviceGlobalFillEventAsCompleted) {
    EXPECT_FALSE(DeviceGlobalFillEventFound)
        << "DeviceGlobalFillEvent was in event wait list but was not expected. "
           "Kernel call "
        << KernelCallCounter;
  } else {
    EXPECT_TRUE(DeviceGlobalFillEventFound)
        << "DeviceGlobalFillEvent expected in event wait list but was missing. "
           "Kernel call "
        << KernelCallCounter;
  }

  bool DeviceGlobalWriteEventFound =
      std::find(event_wait_list, EventListEnd, *DeviceGlobalWriteEvent) !=
      EventListEnd;
  if (TreatDeviceGlobalWriteEventAsCompleted) {
    EXPECT_FALSE(DeviceGlobalWriteEventFound)
        << "DeviceGlobalWriteEvent was in event wait list but was not "
           "expected. Kernel call "
        << KernelCallCounter;
  } else {
    EXPECT_TRUE(DeviceGlobalWriteEventFound)
        << "DeviceGlobalWriteEvent expected in event wait list but was "
           "missing. Kernel call "
        << KernelCallCounter;
  }
  return PI_SUCCESS;
}

} // namespace

void ResetTrackersAndMarkers() {
  DeviceGlobalWriteEvent = std::nullopt;
  DeviceGlobalFillEvent = std::nullopt;
  KernelCallCounter = 0;
  TreatDeviceGlobalFillEventAsCompleted = false;
  TreatDeviceGlobalWriteEventAsCompleted = false;
}

TEST(DeviceGlobalTest, DeviceGlobalInitBeforeUse) {
  ResetTrackersAndMarkers();

  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();

  Mock.redefineAfter<sycl::detail::PiApiKind::piextUSMEnqueueMemset>(
      after_piextUSMEnqueueMemset);
  Mock.redefineAfter<
      sycl::detail::PiApiKind::piextEnqueueDeviceGlobalVariableWrite>(
      after_piextEnqueueDeviceGlobalVariableWrite);
  Mock.redefineAfter<sycl::detail::PiApiKind::piEventGetInfo>(
      after_piEventGetInfo);
  Mock.redefineAfter<sycl::detail::PiApiKind::piEnqueueKernelLaunch>(
      after_piEnqueueKernelLaunch);

  sycl::queue Q{Plt.get_devices()[0]};

  // Kernel call 1.
  // First launch should create both init events.
  Q.single_task<DeviceGlobalTestKernel>([]() {});

  // Kernel call 2.
  // Second launch should have the init events created. If they have not
  // finished they should still be in there.
  Q.single_task<DeviceGlobalTestKernel>([]() {});

  // Kernel call 3.
  // Treat the set init event as finished.
  TreatDeviceGlobalWriteEventAsCompleted = true;
  Q.single_task<DeviceGlobalTestKernel>([]() {});

  // Kernel call 4.
  // Treat the both init event as finished.
  TreatDeviceGlobalFillEventAsCompleted = true;
  Q.single_task<DeviceGlobalTestKernel>([]() {});
}
