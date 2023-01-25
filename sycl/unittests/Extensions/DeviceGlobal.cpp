//==---------------------- DeviceGlobal.cpp --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/pi.hpp>
#include <sycl/sycl.hpp>

#include "detail/context_impl.hpp"
#include "detail/kernel_program_cache.hpp"

#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <optional>

using sycl::detail::PiApiKind;

class DeviceGlobalTestKernel;
constexpr const char *DeviceGlobalTestKernelName = "DeviceGlobalTestKernel";
constexpr const char *DeviceGlobalName = "DeviceGlobalName";
class DeviceGlobalImgScopeTestKernel;
constexpr const char *DeviceGlobalImgScopeTestKernelName =
    "DeviceGlobalImgScopeTestKernel";
constexpr const char *DeviceGlobalImgScopeName = "DeviceGlobalImgScopeName";

sycl::ext::oneapi::experimental::device_global<int[2]> DeviceGlobal;
sycl::ext::oneapi::experimental::device_global<
    int[2], decltype(sycl::ext::oneapi::experimental::properties(
                sycl::ext::oneapi::experimental::device_image_scope))>
    DeviceGlobalImgScope;

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
template <> struct KernelInfo<DeviceGlobalImgScopeTestKernel> {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() {
    return DeviceGlobalImgScopeTestKernelName;
  }
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
      makeDeviceGlobalInfo(DeviceGlobalName, sizeof(int) * 2, 0);
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

static sycl::unittest::PiImage generateDeviceGlobalImgScopeImage() {
  using namespace sycl::unittest;

  // Call device global map initializer explicitly to mimic the integration
  // header.
  sycl::detail::device_global_map::add(&DeviceGlobalImgScope,
                                       DeviceGlobalImgScopeName);

  // Insert remaining device global info into the binary.
  PiPropertySet PropSet;
  PiProperty DevGlobInfo =
      makeDeviceGlobalInfo(DeviceGlobalImgScopeName, sizeof(int) * 2, 1);
  PropSet.insert(__SYCL_PI_PROPERTY_SET_SYCL_DEVICE_GLOBALS,
                 PiArray<PiProperty>{std::move(DevGlobInfo)});

  std::vector<unsigned char> Bin{10, 11, 12, 13, 14, 15}; // Random data

  PiArray<PiOffloadEntry> Entries =
      makeEmptyKernels({DeviceGlobalImgScopeTestKernelName});

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
sycl::unittest::PiImage Imgs[] = {generateDeviceGlobalImage(),
                                  generateDeviceGlobalImgScopeImage()};
sycl::unittest::PiImageArray<2> ImgArray{Imgs};

// Trackers.
thread_local std::optional<pi_event> DeviceGlobalFillEvent = std::nullopt;
thread_local std::optional<pi_event> DeviceGlobalWriteEvent = std::nullopt;
thread_local unsigned KernelCallCounter = 0;
thread_local unsigned DeviceGlobalWriteCounter = 0;
thread_local unsigned DeviceGlobalReadCounter = 0;

// Markers.
thread_local bool TreatDeviceGlobalFillEventAsCompleted = false;
thread_local bool TreatDeviceGlobalWriteEventAsCompleted = false;
thread_local std::optional<pi_program> ExpectedReadWritePIProgram =
    std::nullopt;

static pi_result after_piextUSMEnqueueMemset(pi_queue, void *, pi_int32, size_t,
                                             pi_uint32, const pi_event *,
                                             pi_event *event) {
  EXPECT_FALSE(DeviceGlobalFillEvent.has_value())
      << "piextUSMEnqueueMemset is called multiple times!";
  DeviceGlobalFillEvent = *event;
  return PI_SUCCESS;
}

template <bool Exclusive>
pi_result after_piextEnqueueDeviceGlobalVariableWrite(
    pi_queue, pi_program program, const char *, pi_bool, size_t, size_t,
    const void *, pi_uint32, const pi_event *, pi_event *event) {
  if constexpr (Exclusive) {
    EXPECT_FALSE(DeviceGlobalWriteEvent.has_value())
        << "piextEnqueueDeviceGlobalVariableWrite is called multiple times!";
  }
  if (ExpectedReadWritePIProgram.has_value()) {
    EXPECT_EQ(*ExpectedReadWritePIProgram, program)
        << "piextEnqueueDeviceGlobalVariableWrite did not receive the expected "
           "program!";
  }
  DeviceGlobalWriteEvent = *event;
  ++DeviceGlobalWriteCounter;
  return PI_SUCCESS;
}

pi_result after_piextEnqueueDeviceGlobalVariableRead(
    pi_queue, pi_program program, const char *, pi_bool, size_t, size_t, void *,
    pi_uint32, const pi_event *, pi_event *) {
  if (ExpectedReadWritePIProgram.has_value()) {
    EXPECT_EQ(*ExpectedReadWritePIProgram, program)
        << "piextEnqueueDeviceGlobalVariableRead did not receive the expected "
           "program!";
  }
  ++DeviceGlobalReadCounter;
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

void ResetTrackersAndMarkers() {
  DeviceGlobalWriteEvent = std::nullopt;
  DeviceGlobalFillEvent = std::nullopt;
  KernelCallCounter = 0;
  DeviceGlobalWriteCounter = 0;
  DeviceGlobalReadCounter = 0;
  TreatDeviceGlobalFillEventAsCompleted = false;
  TreatDeviceGlobalWriteEventAsCompleted = false;
  ExpectedReadWritePIProgram = std::nullopt;
}

std::pair<sycl::unittest::PiMock, sycl::queue>
CommonSetup(std::function<void(sycl::unittest::PiMock &)> RedefinitionFunc) {
  ResetTrackersAndMarkers();

  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();

  RedefinitionFunc(Mock);

  // Create new context to isolate device_global initialization.
  sycl::context C{Plt.get_devices()[0]};
  sycl::queue Q{C, Plt.get_devices()[0]};

  return std::make_pair(std::move(Mock), std::move(Q));
}

} // namespace

// Macros for common redefinition calls.
#define REDEFINE_AFTER(API) redefineAfter<PiApiKind::API>(after_##API)
#define REDEFINE_AFTER_TEMPLATED(API, ...)                                     \
  redefineAfter<PiApiKind::API>(after_##API<__VA_ARGS__>)

TEST(DeviceGlobalTest, DeviceGlobalInitBeforeUse) {
  auto [Mock, Q] = CommonSetup([](sycl::unittest::PiMock &MockRef) {
    MockRef.REDEFINE_AFTER(piextUSMEnqueueMemset);
    MockRef.REDEFINE_AFTER_TEMPLATED(piextEnqueueDeviceGlobalVariableWrite,
                                     true);
    MockRef.REDEFINE_AFTER(piEventGetInfo);
    MockRef.REDEFINE_AFTER(piEnqueueKernelLaunch);
  });

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

TEST(DeviceGlobalTest, DeviceGlobalCopyToBeforeUseFull) {
  auto [Mock, Q] = CommonSetup([](sycl::unittest::PiMock &MockRef) {
    MockRef.REDEFINE_AFTER(piextUSMEnqueueMemset);
    MockRef.REDEFINE_AFTER_TEMPLATED(piextEnqueueDeviceGlobalVariableWrite,
                                     true);
    MockRef.REDEFINE_AFTER(piEventGetInfo);
  });

  int Vals[2] = {42, 1234};
  Q.copy(Vals, DeviceGlobal).wait();

  // Device global should not have been written to yet. Fill operation is not
  // needed since the device_global will have a new value.
  EXPECT_TRUE(!DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(!DeviceGlobalFillEvent.has_value());

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();

  // The device global should now have its USM memory pointer written, but fill
  // should still not have happened as an explicit write have happened to the
  // underlying memory.
  EXPECT_TRUE(DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(!DeviceGlobalFillEvent.has_value());
}

TEST(DeviceGlobalTest, DeviceGlobalMemcpyToBeforeUseFull) {
  auto [Mock, Q] = CommonSetup([](sycl::unittest::PiMock &MockRef) {
    MockRef.REDEFINE_AFTER(piextUSMEnqueueMemset);
    MockRef.REDEFINE_AFTER_TEMPLATED(piextEnqueueDeviceGlobalVariableWrite,
                                     true);
    MockRef.REDEFINE_AFTER(piEventGetInfo);
  });

  int Vals[2] = {42, 1234};
  Q.memcpy(DeviceGlobal, Vals).wait();

  // Device global should not have been written to yet. Fill operation is not
  // needed since the device_global will have a new value.
  EXPECT_TRUE(!DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(!DeviceGlobalFillEvent.has_value());

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();

  // The device global should now have its USM memory pointer written, but fill
  // should still not have happened as an explicit write have happened to the
  // underlying memory.
  EXPECT_TRUE(DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(!DeviceGlobalFillEvent.has_value());
}

TEST(DeviceGlobalTest, DeviceGlobalCopyToBeforeUsePartialNoOffset) {
  auto [Mock, Q] = CommonSetup([](sycl::unittest::PiMock &MockRef) {
    MockRef.REDEFINE_AFTER(piextUSMEnqueueMemset);
    MockRef.REDEFINE_AFTER_TEMPLATED(piextEnqueueDeviceGlobalVariableWrite,
                                     true);
    MockRef.REDEFINE_AFTER(piEventGetInfo);
  });

  int Val = 42;
  Q.copy(&Val, DeviceGlobal, 1).wait();

  // Device global should not have been written to yet. The fill operation must
  // have happened as the copy was only partial.
  EXPECT_TRUE(!DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(DeviceGlobalFillEvent.has_value());

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();

  // The device global should now have its USM memory pointer written.
  EXPECT_TRUE(DeviceGlobalWriteEvent.has_value());
}

TEST(DeviceGlobalTest, DeviceGlobalMemcpyToBeforeUsePartialNoOffset) {
  auto [Mock, Q] = CommonSetup([](sycl::unittest::PiMock &MockRef) {
    MockRef.REDEFINE_AFTER(piextUSMEnqueueMemset);
    MockRef.REDEFINE_AFTER_TEMPLATED(piextEnqueueDeviceGlobalVariableWrite,
                                     true);
    MockRef.REDEFINE_AFTER(piEventGetInfo);
  });

  int Val = 42;
  Q.memcpy(DeviceGlobal, &Val, sizeof(int)).wait();

  // Device global should not have been written to yet. The fill operation must
  // have happened as the copy was only partial.
  EXPECT_TRUE(!DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(DeviceGlobalFillEvent.has_value());

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();

  // The device global should now have its USM memory pointer written.
  EXPECT_TRUE(DeviceGlobalWriteEvent.has_value());
}

TEST(DeviceGlobalTest, DeviceGlobalCopyToBeforeUsePartialWithOffset) {
  auto [Mock, Q] = CommonSetup([](sycl::unittest::PiMock &MockRef) {
    MockRef.REDEFINE_AFTER(piextUSMEnqueueMemset);
    MockRef.REDEFINE_AFTER_TEMPLATED(piextEnqueueDeviceGlobalVariableWrite,
                                     true);
    MockRef.REDEFINE_AFTER(piEventGetInfo);
  });

  int Val = 42;
  Q.copy(&Val, DeviceGlobal, 1, 1).wait();

  // Device global should not have been written to yet. The fill operation must
  // have happened as the copy was only partial.
  EXPECT_TRUE(!DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(DeviceGlobalFillEvent.has_value());

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();

  // The device global should now have its USM memory pointer written.
  EXPECT_TRUE(DeviceGlobalWriteEvent.has_value());
}

TEST(DeviceGlobalTest, DeviceGlobalInitBeforeMemcpyToPartialWithOffset) {
  auto [Mock, Q] = CommonSetup([](sycl::unittest::PiMock &MockRef) {
    MockRef.REDEFINE_AFTER(piextUSMEnqueueMemset);
    MockRef.REDEFINE_AFTER_TEMPLATED(piextEnqueueDeviceGlobalVariableWrite,
                                     true);
    MockRef.REDEFINE_AFTER(piEventGetInfo);
  });

  int Val = 42;
  Q.memcpy(DeviceGlobal, &Val, sizeof(int), sizeof(int)).wait();

  // Device global should not have been written to yet. The fill operation must
  // have happened as the copy was only partial.
  EXPECT_TRUE(!DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(DeviceGlobalFillEvent.has_value());

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();

  // The device global should now have its USM memory pointer written.
  EXPECT_TRUE(DeviceGlobalWriteEvent.has_value());
}

TEST(DeviceGlobalTest, DeviceGlobalCopyFromBeforeUse) {
  auto [Mock, Q] = CommonSetup([](sycl::unittest::PiMock &MockRef) {
    MockRef.REDEFINE_AFTER(piextUSMEnqueueMemset);
    MockRef.REDEFINE_AFTER_TEMPLATED(piextEnqueueDeviceGlobalVariableWrite,
                                     true);
    MockRef.REDEFINE_AFTER(piEventGetInfo);
  });

  int Vals[2] = {42, 1234};
  Q.copy(DeviceGlobal, Vals).wait();

  EXPECT_TRUE(!DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(DeviceGlobalFillEvent.has_value());
}

TEST(DeviceGlobalTest, DeviceGlobalMemcpyFromBeforeUse) {
  auto [Mock, Q] = CommonSetup([](sycl::unittest::PiMock &MockRef) {
    MockRef.REDEFINE_AFTER(piextUSMEnqueueMemset);
    MockRef.REDEFINE_AFTER_TEMPLATED(piextEnqueueDeviceGlobalVariableWrite,
                                     true);
    MockRef.REDEFINE_AFTER(piEventGetInfo);
  });

  int Vals[2] = {42, 1234};
  Q.memcpy(Vals, DeviceGlobal).wait();

  EXPECT_TRUE(!DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(DeviceGlobalFillEvent.has_value());
}

TEST(DeviceGlobalTest, DeviceGlobalUseBeforeCopyTo) {
  auto [Mock, Q] = CommonSetup([](sycl::unittest::PiMock &MockRef) {
    MockRef.REDEFINE_AFTER(piextUSMEnqueueMemset);
    MockRef.REDEFINE_AFTER_TEMPLATED(piextEnqueueDeviceGlobalVariableWrite,
                                     true);
    MockRef.REDEFINE_AFTER(piEventGetInfo);
  });

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();

  // Device global will have been initialized at this point.
  EXPECT_TRUE(DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(DeviceGlobalFillEvent.has_value());

  int Vals[2] = {42, 1234};
  Q.copy(Vals, DeviceGlobal).wait();

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();
}

TEST(DeviceGlobalTest, DeviceGlobalUseBeforeMemcpyTo) {
  auto [Mock, Q] = CommonSetup([](sycl::unittest::PiMock &MockRef) {
    MockRef.REDEFINE_AFTER(piextUSMEnqueueMemset);
    MockRef.REDEFINE_AFTER_TEMPLATED(piextEnqueueDeviceGlobalVariableWrite,
                                     true);
    MockRef.REDEFINE_AFTER(piEventGetInfo);
  });

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();

  // Device global will have been initialized at this point.
  EXPECT_TRUE(DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(DeviceGlobalFillEvent.has_value());

  // Copy to should only copy to USM memory. If fill or a copy to the device
  // global happens it should fail here.
  int Vals[2] = {42, 1234};
  Q.memcpy(DeviceGlobal, Vals).wait();

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();
}

TEST(DeviceGlobalTest, DeviceGlobalImgScopeCopyToBeforeUse) {
  auto [Mock, Q] = CommonSetup([](sycl::unittest::PiMock &MockRef) {
    MockRef.REDEFINE_AFTER_TEMPLATED(piextEnqueueDeviceGlobalVariableWrite,
                                     false);
    MockRef.REDEFINE_AFTER(piextEnqueueDeviceGlobalVariableRead);
  });

  int Vals[2] = {42, 1234};
  Q.copy(Vals, DeviceGlobalImgScope).wait();

  Q.single_task<DeviceGlobalImgScopeTestKernel>([]() {}).wait();

  EXPECT_EQ(DeviceGlobalWriteCounter, 1u);
  EXPECT_EQ(DeviceGlobalReadCounter, 0u);
}

TEST(DeviceGlobalTest, DeviceGlobalImgScopeMemcpyToBeforeUse) {
  auto [Mock, Q] = CommonSetup([](sycl::unittest::PiMock &MockRef) {
    MockRef.REDEFINE_AFTER_TEMPLATED(piextEnqueueDeviceGlobalVariableWrite,
                                     false);
    MockRef.REDEFINE_AFTER(piextEnqueueDeviceGlobalVariableRead);
  });

  int Vals[2] = {42, 1234};
  Q.memcpy(DeviceGlobalImgScope, Vals).wait();

  Q.single_task<DeviceGlobalImgScopeTestKernel>([]() {}).wait();

  EXPECT_EQ(DeviceGlobalWriteCounter, 1u);
  EXPECT_EQ(DeviceGlobalReadCounter, 0u);
}

TEST(DeviceGlobalTest, DeviceGlobalImgScopeCopyFromBeforeUse) {
  auto [Mock, Q] = CommonSetup([](sycl::unittest::PiMock &MockRef) {
    MockRef.REDEFINE_AFTER_TEMPLATED(piextEnqueueDeviceGlobalVariableWrite,
                                     false);
    MockRef.REDEFINE_AFTER(piextEnqueueDeviceGlobalVariableRead);
  });

  int Vals[2] = {42, 1234};
  Q.copy(DeviceGlobalImgScope, Vals).wait();

  Q.single_task<DeviceGlobalImgScopeTestKernel>([]() {}).wait();

  EXPECT_EQ(DeviceGlobalWriteCounter, 0u);
  EXPECT_EQ(DeviceGlobalReadCounter, 1u);
}

TEST(DeviceGlobalTest, DeviceGlobalImgScopeMemcpyFromBeforeUse) {
  auto [Mock, Q] = CommonSetup([](sycl::unittest::PiMock &MockRef) {
    MockRef.REDEFINE_AFTER_TEMPLATED(piextEnqueueDeviceGlobalVariableWrite,
                                     false);
    MockRef.REDEFINE_AFTER(piextEnqueueDeviceGlobalVariableRead);
  });

  int Vals[2] = {42, 1234};
  Q.memcpy(Vals, DeviceGlobalImgScope).wait();

  Q.single_task<DeviceGlobalImgScopeTestKernel>([]() {}).wait();

  EXPECT_EQ(DeviceGlobalWriteCounter, 0u);
  EXPECT_EQ(DeviceGlobalReadCounter, 1u);
}

TEST(DeviceGlobalTest, DeviceGlobalImgScopeUseBeforeCopyTo) {
  auto [Mock, Q] = CommonSetup([](sycl::unittest::PiMock &MockRef) {
    MockRef.REDEFINE_AFTER_TEMPLATED(piextEnqueueDeviceGlobalVariableWrite,
                                     false);
    MockRef.REDEFINE_AFTER(piextEnqueueDeviceGlobalVariableRead);
  });

  Q.single_task<DeviceGlobalImgScopeTestKernel>([]() {}).wait();

  // Register the cached program as expected for device global memory operation.
  auto CtxImpl = sycl::detail::getSyclObjImpl(Q.get_context());
  sycl::detail::KernelProgramCache::KernelCacheT &KernelCache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  ASSERT_EQ(KernelCache.size(), (size_t)1)
      << "Expect 1 program in kernel cache";
  ExpectedReadWritePIProgram = KernelCache.begin()->first;

  // Expect no write or read yet.
  EXPECT_EQ(DeviceGlobalWriteCounter, 0u);
  EXPECT_EQ(DeviceGlobalReadCounter, 0u);

  int Vals[2] = {42, 1234};
  Q.copy(Vals, DeviceGlobalImgScope).wait();

  EXPECT_EQ(DeviceGlobalWriteCounter, 1u);
  EXPECT_EQ(DeviceGlobalReadCounter, 0u);
}

TEST(DeviceGlobalTest, DeviceGlobalImgScopeUseBeforeMemcpyTo) {
  auto [Mock, Q] = CommonSetup([](sycl::unittest::PiMock &MockRef) {
    MockRef.REDEFINE_AFTER_TEMPLATED(piextEnqueueDeviceGlobalVariableWrite,
                                     false);
    MockRef.REDEFINE_AFTER(piextEnqueueDeviceGlobalVariableRead);
  });

  Q.single_task<DeviceGlobalImgScopeTestKernel>([]() {}).wait();

  // Register the cached program as expected for device global memory operation.
  auto CtxImpl = sycl::detail::getSyclObjImpl(Q.get_context());
  sycl::detail::KernelProgramCache::KernelCacheT &KernelCache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  ASSERT_EQ(KernelCache.size(), (size_t)1)
      << "Expect 1 program in kernel cache";
  ExpectedReadWritePIProgram = KernelCache.begin()->first;

  // Expect no write or read yet.
  EXPECT_EQ(DeviceGlobalWriteCounter, 0u);
  EXPECT_EQ(DeviceGlobalReadCounter, 0u);

  int Vals[2] = {42, 1234};
  Q.memcpy(DeviceGlobalImgScope, Vals).wait();

  EXPECT_EQ(DeviceGlobalWriteCounter, 1u);
  EXPECT_EQ(DeviceGlobalReadCounter, 0u);
}

TEST(DeviceGlobalTest, DeviceGlobalImgScopeUseBeforeCopyFrom) {
  auto [Mock, Q] = CommonSetup([](sycl::unittest::PiMock &MockRef) {
    MockRef.REDEFINE_AFTER_TEMPLATED(piextEnqueueDeviceGlobalVariableWrite,
                                     false);
    MockRef.REDEFINE_AFTER(piextEnqueueDeviceGlobalVariableRead);
  });

  Q.single_task<DeviceGlobalImgScopeTestKernel>([]() {}).wait();

  // Register the cached program as expected for device global memory operation.
  auto CtxImpl = sycl::detail::getSyclObjImpl(Q.get_context());
  sycl::detail::KernelProgramCache::KernelCacheT &KernelCache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  ASSERT_EQ(KernelCache.size(), (size_t)1)
      << "Expect 1 program in kernel cache";
  ExpectedReadWritePIProgram = KernelCache.begin()->first;

  // Expect no write or read yet.
  EXPECT_EQ(DeviceGlobalWriteCounter, 0u);
  EXPECT_EQ(DeviceGlobalReadCounter, 0u);

  int Vals[2] = {42, 1234};
  Q.copy(DeviceGlobalImgScope, Vals).wait();

  EXPECT_EQ(DeviceGlobalWriteCounter, 0u);
  EXPECT_EQ(DeviceGlobalReadCounter, 1u);
}

TEST(DeviceGlobalTest, DeviceGlobalImgScopeUseBeforeMemcpyFrom) {
  auto [Mock, Q] = CommonSetup([](sycl::unittest::PiMock &MockRef) {
    MockRef.REDEFINE_AFTER_TEMPLATED(piextEnqueueDeviceGlobalVariableWrite,
                                     false);
    MockRef.REDEFINE_AFTER(piextEnqueueDeviceGlobalVariableRead);
  });

  Q.single_task<DeviceGlobalImgScopeTestKernel>([]() {}).wait();

  // Register the cached program as expected for device global memory operation.
  auto CtxImpl = sycl::detail::getSyclObjImpl(Q.get_context());
  sycl::detail::KernelProgramCache::KernelCacheT &KernelCache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  ASSERT_EQ(KernelCache.size(), (size_t)1)
      << "Expect 1 program in kernel cache";
  ExpectedReadWritePIProgram = KernelCache.begin()->first;

  // Expect no write or read yet.
  EXPECT_EQ(DeviceGlobalWriteCounter, 0u);
  EXPECT_EQ(DeviceGlobalReadCounter, 0u);

  int Vals[2] = {42, 1234};
  Q.memcpy(Vals, DeviceGlobalImgScope).wait();

  EXPECT_EQ(DeviceGlobalWriteCounter, 0u);
  EXPECT_EQ(DeviceGlobalReadCounter, 1u);
}
