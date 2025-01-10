//==---------------------- DeviceGlobal.cpp --------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include "detail/context_impl.hpp"
#include "detail/kernel_program_cache.hpp"

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

class DeviceGlobalTestKernel;
constexpr const char *DeviceGlobalTestKernelName = "DeviceGlobalTestKernel";
constexpr const char *DeviceGlobalName = "DeviceGlobalName";
class DeviceGlobalImgScopeTestKernel;
constexpr const char *DeviceGlobalImgScopeTestKernelName =
    "DeviceGlobalImgScopeTestKernel";
constexpr const char *DeviceGlobalImgScopeName = "DeviceGlobalImgScopeName";

using DeviceGlobalElemType = int[2];
sycl::ext::oneapi::experimental::device_global<DeviceGlobalElemType>
    DeviceGlobal;
sycl::ext::oneapi::experimental::device_global<
    DeviceGlobalElemType,
    decltype(sycl::ext::oneapi::experimental::properties(
        sycl::ext::oneapi::experimental::device_image_scope))>
    DeviceGlobalImgScope;

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<DeviceGlobalTestKernel>
    : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return DeviceGlobalTestKernelName; }
};
template <>
struct KernelInfo<DeviceGlobalImgScopeTestKernel>
    : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() {
    return DeviceGlobalImgScopeTestKernelName;
  }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::MockDeviceImage generateDeviceGlobalImage() {
  using namespace sycl::unittest;

  // Call device global map initializer explicitly to mimic the integration
  // header.
  sycl::detail::device_global_map::add(&DeviceGlobal, DeviceGlobalName);

  // Insert remaining device global info into the binary.
  MockPropertySet PropSet;
  MockProperty DevGlobInfo =
      makeDeviceGlobalInfo(DeviceGlobalName, sizeof(int) * 2, 0);
  PropSet.insert(__SYCL_PROPERTY_SET_SYCL_DEVICE_GLOBALS,
                 std::vector<MockProperty>{std::move(DevGlobInfo)});

  std::vector<MockOffloadEntry> Entries =
      makeEmptyKernels({DeviceGlobalTestKernelName});

  MockDeviceImage Img(std::move(Entries), std::move(PropSet));

  return Img;
}

static sycl::unittest::MockDeviceImage generateDeviceGlobalImgScopeImage() {
  using namespace sycl::unittest;

  // Call device global map initializer explicitly to mimic the integration
  // header.
  sycl::detail::device_global_map::add(&DeviceGlobalImgScope,
                                       DeviceGlobalImgScopeName);

  // Insert remaining device global info into the binary.
  MockPropertySet PropSet;
  MockProperty DevGlobInfo =
      makeDeviceGlobalInfo(DeviceGlobalImgScopeName, sizeof(int) * 2, 1);
  PropSet.insert(__SYCL_PROPERTY_SET_SYCL_DEVICE_GLOBALS,
                 std::vector<MockProperty>{std::move(DevGlobInfo)});

  std::vector<MockOffloadEntry> Entries =
      makeEmptyKernels({DeviceGlobalImgScopeTestKernelName});

  MockDeviceImage Img(std::move(Entries), std::move(PropSet));

  return Img;
}

namespace {
sycl::unittest::MockDeviceImage Imgs[] = {generateDeviceGlobalImage(),
                                          generateDeviceGlobalImgScopeImage()};
sycl::unittest::MockDeviceImageArray<2> ImgArray{Imgs};

// Trackers.
thread_local DeviceGlobalElemType MockDeviceGlobalMem;
thread_local DeviceGlobalElemType MockDeviceGlobalImgScopeMem;
thread_local std::optional<ur_event_handle_t> DeviceGlobalInitEvent =
    std::nullopt;
thread_local std::optional<ur_event_handle_t> DeviceGlobalWriteEvent =
    std::nullopt;
thread_local unsigned KernelCallCounter = 0;
thread_local unsigned DeviceGlobalWriteCounter = 0;
thread_local unsigned DeviceGlobalReadCounter = 0;

// Markers.
thread_local bool TreatDeviceGlobalInitEventAsCompleted = false;
thread_local bool TreatDeviceGlobalWriteEventAsCompleted = false;
thread_local std::optional<ur_program_handle_t> ExpectedReadWriteURProgram =
    std::nullopt;

static ur_result_t after_urUSMDeviceAlloc(void *pParams) {
  auto params = *static_cast<ur_usm_device_alloc_params_t *>(pParams);
  // Use the mock memory.
  **params.pppMem = MockDeviceGlobalMem;
  return UR_RESULT_SUCCESS;
}

static ur_result_t after_urEnqueueUSMMemcpy(void *pParams) {
  auto params = *static_cast<ur_enqueue_usm_memcpy_params_t *>(pParams);
  // If DeviceGlobalInitEvent.has_value() is true then this means that this is
  // the second call to MemCopy and we don't want to initialize anything. If
  // it's the first call then we want to set the DeviceGlobalInitEvent
  if (!DeviceGlobalInitEvent.has_value())
    DeviceGlobalInitEvent = **params.pphEvent;
  std::memcpy(*params.ppDst, *params.ppSrc, *params.psize);
  return UR_RESULT_SUCCESS;
}

template <bool Exclusive>
ur_result_t after_urEnqueueDeviceGlobalVariableWrite(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_device_global_variable_write_params_t *>(pParams);
  if constexpr (Exclusive) {
    EXPECT_FALSE(DeviceGlobalWriteEvent.has_value())
        << "urEnqueueDeviceGlobalVariableWrite is called multiple times!";
  }
  if (ExpectedReadWriteURProgram.has_value()) {
    EXPECT_EQ(*ExpectedReadWriteURProgram, *params.phProgram)
        << "urEnqueueDeviceGlobalVariableWrite did not receive the expected "
           "program!";
  }
  std::memcpy(MockDeviceGlobalImgScopeMem + *params.poffset, *params.ppSrc,
              *params.pcount);
  DeviceGlobalWriteEvent = **params.pphEvent;
  ++DeviceGlobalWriteCounter;
  return UR_RESULT_SUCCESS;
}

ur_result_t after_urEnqueueDeviceGlobalVariableRead(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_device_global_variable_read_params_t *>(pParams);
  if (ExpectedReadWriteURProgram.has_value()) {
    EXPECT_EQ(*ExpectedReadWriteURProgram, *params.phProgram)
        << "urEnqueueDeviceGlobalVariableRead did not receive the expected "
           "program!";
  }
  std::memcpy(*params.ppDst, MockDeviceGlobalImgScopeMem + *params.poffset,
              *params.pcount);
  ++DeviceGlobalReadCounter;
  return UR_RESULT_SUCCESS;
}

ur_result_t after_urEventGetInfo(void *pParams) {
  auto params = *static_cast<ur_event_get_info_params_t *>(pParams);
  if (*params.ppropName == UR_EVENT_INFO_COMMAND_EXECUTION_STATUS &&
      *params.ppPropValue != nullptr) {
    if ((TreatDeviceGlobalInitEventAsCompleted &&
         DeviceGlobalInitEvent.has_value() &&
         *params.phEvent == *DeviceGlobalInitEvent) ||
        (TreatDeviceGlobalWriteEventAsCompleted &&
         DeviceGlobalWriteEvent.has_value() &&
         *params.phEvent == *DeviceGlobalWriteEvent))
      *static_cast<ur_event_status_t *>(*params.ppPropValue) =
          UR_EVENT_STATUS_COMPLETE;
    else
      *static_cast<ur_event_status_t *>(*params.ppPropValue) =
          UR_EVENT_STATUS_SUBMITTED;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t after_urEnqueueKernelLaunch(void *pParams) {
  auto params = *static_cast<ur_enqueue_kernel_launch_params_t *>(pParams);
  ++KernelCallCounter;
  EXPECT_TRUE(DeviceGlobalInitEvent.has_value())
      << "DeviceGlobalInitEvent has not been set. Kernel call "
      << KernelCallCounter;
  EXPECT_TRUE(DeviceGlobalWriteEvent.has_value())
      << "DeviceGlobalWriteEvent has not been set. Kernel call "
      << KernelCallCounter;

  const ur_event_handle_t *EventListEnd =
      *params.pphEventWaitList + *params.pnumEventsInWaitList;

  bool DeviceGlobalInitEventFound =
      std::find(*params.pphEventWaitList, EventListEnd,
                *DeviceGlobalInitEvent) != EventListEnd;
  if (TreatDeviceGlobalInitEventAsCompleted) {
    EXPECT_FALSE(DeviceGlobalInitEventFound)
        << "DeviceGlobalInitEvent was in event wait list but was not expected. "
           "Kernel call "
        << KernelCallCounter;
  } else {
    EXPECT_TRUE(DeviceGlobalInitEventFound)
        << "DeviceGlobalInitEvent expected in event wait list but was missing. "
           "Kernel call "
        << KernelCallCounter;
  }

  bool DeviceGlobalWriteEventFound =
      std::find(*params.pphEventWaitList, EventListEnd,
                *DeviceGlobalWriteEvent) != EventListEnd;
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
  return UR_RESULT_SUCCESS;
}

} // namespace

class DeviceGlobalTest : public ::testing::Test {
  void SetUp() {
    ResetTrackersAndMarkers();
    sycl::platform Plt = sycl::platform();
    sycl::context C{Plt.get_devices()[0]};
    Q = sycl::queue(C, Plt.get_devices()[0]);
  }

  void ResetTrackersAndMarkers() {
    std::memset(MockDeviceGlobalMem, 1, sizeof(DeviceGlobalElemType));
    std::memset(MockDeviceGlobalImgScopeMem, 0, sizeof(DeviceGlobalElemType));
    DeviceGlobalWriteEvent = std::nullopt;
    DeviceGlobalInitEvent = std::nullopt;
    KernelCallCounter = 0;
    DeviceGlobalWriteCounter = 0;
    DeviceGlobalReadCounter = 0;
    TreatDeviceGlobalInitEventAsCompleted = false;
    TreatDeviceGlobalWriteEventAsCompleted = false;
    ExpectedReadWriteURProgram = std::nullopt;
  }

public:
  sycl::unittest::UrMock<> Mock;
  sycl::queue Q;
};

// Macros for common redefinition calls.
#define REDEFINE_AFTER(API)                                                    \
  mock::getCallbacks().set_after_callback(#API, &after_##API)
#define REDEFINE_AFTER_TEMPLATED(API, ...)                                     \
  mock::getCallbacks().set_after_callback(#API, &after_##API<__VA_ARGS__>)

TEST_F(DeviceGlobalTest, DeviceGlobalInitBeforeUse) {
  REDEFINE_AFTER(urUSMDeviceAlloc);
  REDEFINE_AFTER(urEnqueueUSMMemcpy);
  REDEFINE_AFTER_TEMPLATED(urEnqueueDeviceGlobalVariableWrite, true);
  REDEFINE_AFTER(urEventGetInfo);
  REDEFINE_AFTER(urEnqueueKernelLaunch);

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
  TreatDeviceGlobalInitEventAsCompleted = true;
  Q.single_task<DeviceGlobalTestKernel>([]() {});
}

TEST_F(DeviceGlobalTest, DeviceGlobalInitialMemContents) {
  REDEFINE_AFTER(urUSMDeviceAlloc);
  REDEFINE_AFTER(urEnqueueUSMMemcpy);
  REDEFINE_AFTER(urEnqueueDeviceGlobalVariableRead);

  int Results[2] = {3, 4};
  // This should replace the contents of Results with {0, 0}
  Q.copy(DeviceGlobal, Results).wait();

  // Device global should not have been read from yet. Memcpy operation is
  // required to init for full copies as certain orderings could get invalid
  // reads otherwise.
  EXPECT_TRUE(!DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(DeviceGlobalInitEvent.has_value());

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();

  // Check the mocked memory.
  EXPECT_EQ(MockDeviceGlobalMem[0], Results[0]);
  EXPECT_EQ(MockDeviceGlobalMem[1], Results[1]);
}

TEST_F(DeviceGlobalTest, DeviceGlobalCopyToBeforeUseFull) {
  REDEFINE_AFTER(urUSMDeviceAlloc);
  REDEFINE_AFTER(urEnqueueUSMMemcpy);
  REDEFINE_AFTER_TEMPLATED(urEnqueueDeviceGlobalVariableWrite, true);
  REDEFINE_AFTER(urEventGetInfo);

  int Vals[2] = {42, 1234};
  Q.copy(Vals, DeviceGlobal).wait();

  // Device global should not have been written to yet. Memcpy operation is
  // required to init for full copies as certain orderings could get invalid
  // reads otherwise.
  EXPECT_TRUE(!DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(DeviceGlobalInitEvent.has_value());

  // Check the mocked memory.
  EXPECT_EQ(MockDeviceGlobalMem[0], Vals[0]);
  EXPECT_EQ(MockDeviceGlobalMem[1], Vals[1]);

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();

  // The device global should now have its USM memory pointer written.
  EXPECT_TRUE(DeviceGlobalWriteEvent.has_value());
}

TEST_F(DeviceGlobalTest, DeviceGlobalMemcpyToBeforeUseFull) {
  REDEFINE_AFTER(urUSMDeviceAlloc);
  REDEFINE_AFTER(urEnqueueUSMMemcpy);
  REDEFINE_AFTER_TEMPLATED(urEnqueueDeviceGlobalVariableWrite, true);
  REDEFINE_AFTER(urEventGetInfo);

  int Vals[2] = {42, 1234};
  Q.memcpy(DeviceGlobal, Vals).wait();

  // Device global should not have been written to yet. Memcpy operation is
  // required to init for full copies as certain orderings could get invalid
  // reads otherwise.
  EXPECT_TRUE(!DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(DeviceGlobalInitEvent.has_value());

  // Check the mocked memory.
  EXPECT_EQ(MockDeviceGlobalMem[0], Vals[0]);
  EXPECT_EQ(MockDeviceGlobalMem[1], Vals[1]);

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();

  // The device global should now have its USM memory pointer written.
  EXPECT_TRUE(DeviceGlobalWriteEvent.has_value());
}

TEST_F(DeviceGlobalTest, DeviceGlobalCopyToBeforeUsePartialNoOffset) {
  REDEFINE_AFTER(urUSMDeviceAlloc);
  REDEFINE_AFTER(urEnqueueUSMMemcpy);
  REDEFINE_AFTER_TEMPLATED(urEnqueueDeviceGlobalVariableWrite, true);
  REDEFINE_AFTER(urEventGetInfo);

  int Val = 42;
  Q.copy(&Val, DeviceGlobal, 1).wait();

  // Device global should not have been written to yet. The Memcpy operation (to
  // initialize the memory) must have happened as the copy was only partial.
  EXPECT_TRUE(!DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(DeviceGlobalInitEvent.has_value());

  // Check the mocked memory.
  EXPECT_EQ(MockDeviceGlobalMem[0], Val);
  EXPECT_EQ(MockDeviceGlobalMem[1], 0);

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();

  // The device global should now have its USM memory pointer written.
  EXPECT_TRUE(DeviceGlobalWriteEvent.has_value());
}

TEST_F(DeviceGlobalTest, DeviceGlobalMemcpyToBeforeUsePartialNoOffset) {
  REDEFINE_AFTER(urUSMDeviceAlloc);
  REDEFINE_AFTER(urEnqueueUSMMemcpy);
  REDEFINE_AFTER_TEMPLATED(urEnqueueDeviceGlobalVariableWrite, true);
  REDEFINE_AFTER(urEventGetInfo);

  int Val = 42;
  Q.memcpy(DeviceGlobal, &Val, sizeof(int)).wait();

  // Device global should not have been written to yet. The Memcpy operation
  // required to init must have happened as the copy was only partial.
  EXPECT_TRUE(!DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(DeviceGlobalInitEvent.has_value());

  // Check the mocked memory.
  EXPECT_EQ(MockDeviceGlobalMem[0], Val);
  EXPECT_EQ(MockDeviceGlobalMem[1], 0);

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();

  // The device global should now have its USM memory pointer written.
  EXPECT_TRUE(DeviceGlobalWriteEvent.has_value());
}

TEST_F(DeviceGlobalTest, DeviceGlobalCopyToBeforeUsePartialWithOffset) {
  REDEFINE_AFTER(urUSMDeviceAlloc);
  REDEFINE_AFTER(urEnqueueUSMMemcpy);
  REDEFINE_AFTER_TEMPLATED(urEnqueueDeviceGlobalVariableWrite, true);
  REDEFINE_AFTER(urEventGetInfo);

  int Val = 42;
  Q.copy(&Val, DeviceGlobal, 1, 1).wait();

  // Device global should not have been written to yet. The Memcopy operation
  // required to init must have happened as the copy was only partial.
  EXPECT_TRUE(!DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(DeviceGlobalInitEvent.has_value());

  // Check the mocked memory.
  EXPECT_EQ(MockDeviceGlobalMem[0], 0);
  EXPECT_EQ(MockDeviceGlobalMem[1], Val);

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();

  // The device global should now have its USM memory pointer written.
  EXPECT_TRUE(DeviceGlobalWriteEvent.has_value());
}

TEST_F(DeviceGlobalTest, DeviceGlobalInitBeforeMemcpyToPartialWithOffset) {
  REDEFINE_AFTER(urUSMDeviceAlloc);
  REDEFINE_AFTER(urEnqueueUSMMemcpy);
  REDEFINE_AFTER_TEMPLATED(urEnqueueDeviceGlobalVariableWrite, true);
  REDEFINE_AFTER(urEventGetInfo);

  int Val = 42;
  Q.memcpy(DeviceGlobal, &Val, sizeof(int), sizeof(int)).wait();

  // Device global should not have been written to yet. The Memcpy operation
  // required to init must have happened as the copy was only partial.
  EXPECT_TRUE(!DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(DeviceGlobalInitEvent.has_value());

  // Check the mocked memory.
  EXPECT_EQ(MockDeviceGlobalMem[0], 0);
  EXPECT_EQ(MockDeviceGlobalMem[1], Val);

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();

  // The device global should now have its USM memory pointer written.
  EXPECT_TRUE(DeviceGlobalWriteEvent.has_value());
}

TEST_F(DeviceGlobalTest, DeviceGlobalCopyFromBeforeUse) {
  REDEFINE_AFTER(urUSMDeviceAlloc);
  REDEFINE_AFTER(urEnqueueUSMMemcpy);
  REDEFINE_AFTER_TEMPLATED(urEnqueueDeviceGlobalVariableWrite, true);
  REDEFINE_AFTER(urEventGetInfo);

  int Vals[2] = {42, 1234};
  Q.copy(DeviceGlobal, Vals).wait();

  EXPECT_TRUE(!DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(DeviceGlobalInitEvent.has_value());

  // Check the mocked memory.
  EXPECT_EQ(MockDeviceGlobalMem[0], Vals[0]);
  EXPECT_EQ(MockDeviceGlobalMem[1], Vals[1]);
}

TEST_F(DeviceGlobalTest, DeviceGlobalMemcpyFromBeforeUse) {
  REDEFINE_AFTER(urUSMDeviceAlloc);
  REDEFINE_AFTER(urEnqueueUSMMemcpy);
  REDEFINE_AFTER_TEMPLATED(urEnqueueDeviceGlobalVariableWrite, true);
  REDEFINE_AFTER(urEventGetInfo);

  int Vals[2] = {42, 1234};
  Q.memcpy(Vals, DeviceGlobal).wait();

  EXPECT_TRUE(!DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(DeviceGlobalInitEvent.has_value());

  // Check the mocked memory.
  EXPECT_EQ(MockDeviceGlobalMem[0], Vals[0]);
  EXPECT_EQ(MockDeviceGlobalMem[1], Vals[1]);
}

TEST_F(DeviceGlobalTest, DeviceGlobalUseBeforeCopyTo) {
  REDEFINE_AFTER(urUSMDeviceAlloc);
  REDEFINE_AFTER(urEnqueueUSMMemcpy);
  REDEFINE_AFTER_TEMPLATED(urEnqueueDeviceGlobalVariableWrite, true);
  REDEFINE_AFTER(urEventGetInfo);

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();

  // Device global will have been initialized at this point.
  EXPECT_TRUE(DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(DeviceGlobalInitEvent.has_value());

  int Vals[2] = {42, 1234};
  Q.copy(Vals, DeviceGlobal).wait();

  // Check the mocked memory.
  EXPECT_EQ(MockDeviceGlobalMem[0], Vals[0]);
  EXPECT_EQ(MockDeviceGlobalMem[1], Vals[1]);

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();
}

TEST_F(DeviceGlobalTest, DeviceGlobalUseBeforeMemcpyTo) {
  REDEFINE_AFTER(urUSMDeviceAlloc);
  REDEFINE_AFTER(urEnqueueUSMMemcpy);
  REDEFINE_AFTER_TEMPLATED(urEnqueueDeviceGlobalVariableWrite, true);
  REDEFINE_AFTER(urEventGetInfo);

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();

  // Device global will have been initialized at this point.
  EXPECT_TRUE(DeviceGlobalWriteEvent.has_value());
  EXPECT_TRUE(DeviceGlobalInitEvent.has_value());

  // Copy to should only copy to USM memory. If fill or a copy to the device
  // global happens it should fail here.
  int Vals[2] = {42, 1234};
  Q.memcpy(DeviceGlobal, Vals).wait();

  // Check the mocked memory.
  EXPECT_EQ(MockDeviceGlobalMem[0], Vals[0]);
  EXPECT_EQ(MockDeviceGlobalMem[1], Vals[1]);

  Q.single_task<DeviceGlobalTestKernel>([]() {}).wait();
}

TEST_F(DeviceGlobalTest, DeviceGlobalImgScopeCopyToBeforeUse) {
  REDEFINE_AFTER_TEMPLATED(urEnqueueDeviceGlobalVariableWrite, false);
  REDEFINE_AFTER(urEnqueueDeviceGlobalVariableRead);

  int Vals[2] = {42, 1234};
  Q.copy(Vals, DeviceGlobalImgScope).wait();

  Q.single_task<DeviceGlobalImgScopeTestKernel>([]() {}).wait();

  EXPECT_EQ(DeviceGlobalWriteCounter, 1u);
  EXPECT_EQ(DeviceGlobalReadCounter, 0u);

  // Check the mocked memory.
  EXPECT_EQ(MockDeviceGlobalImgScopeMem[0], Vals[0]);
  EXPECT_EQ(MockDeviceGlobalImgScopeMem[1], Vals[1]);
}

TEST_F(DeviceGlobalTest, DeviceGlobalImgScopeMemcpyToBeforeUse) {
  REDEFINE_AFTER_TEMPLATED(urEnqueueDeviceGlobalVariableWrite, false);
  REDEFINE_AFTER(urEnqueueDeviceGlobalVariableRead);

  int Vals[2] = {42, 1234};
  Q.memcpy(DeviceGlobalImgScope, Vals).wait();

  Q.single_task<DeviceGlobalImgScopeTestKernel>([]() {}).wait();

  EXPECT_EQ(DeviceGlobalWriteCounter, 1u);
  EXPECT_EQ(DeviceGlobalReadCounter, 0u);

  // Check the mocked memory.
  EXPECT_EQ(MockDeviceGlobalImgScopeMem[0], Vals[0]);
  EXPECT_EQ(MockDeviceGlobalImgScopeMem[1], Vals[1]);
}

TEST_F(DeviceGlobalTest, DeviceGlobalImgScopeCopyFromBeforeUse) {
  REDEFINE_AFTER_TEMPLATED(urEnqueueDeviceGlobalVariableWrite, false);
  REDEFINE_AFTER(urEnqueueDeviceGlobalVariableRead);

  int Vals[2] = {42, 1234};
  Q.copy(DeviceGlobalImgScope, Vals).wait();

  Q.single_task<DeviceGlobalImgScopeTestKernel>([]() {}).wait();

  EXPECT_EQ(DeviceGlobalWriteCounter, 0u);
  EXPECT_EQ(DeviceGlobalReadCounter, 1u);

  // Check the mocked memory.
  EXPECT_EQ(MockDeviceGlobalImgScopeMem[0], Vals[0]);
  EXPECT_EQ(MockDeviceGlobalImgScopeMem[1], Vals[1]);
}

TEST_F(DeviceGlobalTest, DeviceGlobalImgScopeMemcpyFromBeforeUse) {
  REDEFINE_AFTER_TEMPLATED(urEnqueueDeviceGlobalVariableWrite, false);
  REDEFINE_AFTER(urEnqueueDeviceGlobalVariableRead);

  int Vals[2] = {42, 1234};
  Q.memcpy(Vals, DeviceGlobalImgScope).wait();

  Q.single_task<DeviceGlobalImgScopeTestKernel>([]() {}).wait();

  EXPECT_EQ(DeviceGlobalWriteCounter, 0u);
  EXPECT_EQ(DeviceGlobalReadCounter, 1u);

  // Check the mocked memory.
  EXPECT_EQ(MockDeviceGlobalImgScopeMem[0], Vals[0]);
  EXPECT_EQ(MockDeviceGlobalImgScopeMem[1], Vals[1]);
}

TEST_F(DeviceGlobalTest, DeviceGlobalImgScopeUseBeforeCopyTo) {
  REDEFINE_AFTER_TEMPLATED(urEnqueueDeviceGlobalVariableWrite, false);
  REDEFINE_AFTER(urEnqueueDeviceGlobalVariableRead);

  Q.single_task<DeviceGlobalImgScopeTestKernel>([]() {}).wait();

  // Register the cached program as expected for device global memory operation.
  auto CtxImpl = sycl::detail::getSyclObjImpl(Q.get_context());
  sycl::detail::KernelProgramCache::KernelCacheT &KernelCache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  ASSERT_EQ(KernelCache.size(), (size_t)1)
      << "Expect 1 program in kernel cache";
  ExpectedReadWriteURProgram = KernelCache.begin()->first;

  // Expect no write or read yet.
  EXPECT_EQ(DeviceGlobalWriteCounter, 0u);
  EXPECT_EQ(DeviceGlobalReadCounter, 0u);

  int Vals[2] = {42, 1234};
  Q.copy(Vals, DeviceGlobalImgScope).wait();

  EXPECT_EQ(DeviceGlobalWriteCounter, 1u);
  EXPECT_EQ(DeviceGlobalReadCounter, 0u);

  // Check the mocked memory.
  EXPECT_EQ(MockDeviceGlobalImgScopeMem[0], Vals[0]);
  EXPECT_EQ(MockDeviceGlobalImgScopeMem[1], Vals[1]);
}

TEST_F(DeviceGlobalTest, DeviceGlobalImgScopeUseBeforeMemcpyTo) {
  REDEFINE_AFTER_TEMPLATED(urEnqueueDeviceGlobalVariableWrite, false);
  REDEFINE_AFTER(urEnqueueDeviceGlobalVariableRead);

  Q.single_task<DeviceGlobalImgScopeTestKernel>([]() {}).wait();

  // Register the cached program as expected for device global memory operation.
  auto CtxImpl = sycl::detail::getSyclObjImpl(Q.get_context());
  sycl::detail::KernelProgramCache::KernelCacheT &KernelCache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  ASSERT_EQ(KernelCache.size(), (size_t)1)
      << "Expect 1 program in kernel cache";
  ExpectedReadWriteURProgram = KernelCache.begin()->first;

  // Expect no write or read yet.
  EXPECT_EQ(DeviceGlobalWriteCounter, 0u);
  EXPECT_EQ(DeviceGlobalReadCounter, 0u);

  int Vals[2] = {42, 1234};
  Q.memcpy(DeviceGlobalImgScope, Vals).wait();

  EXPECT_EQ(DeviceGlobalWriteCounter, 1u);
  EXPECT_EQ(DeviceGlobalReadCounter, 0u);

  // Check the mocked memory.
  EXPECT_EQ(MockDeviceGlobalImgScopeMem[0], Vals[0]);
  EXPECT_EQ(MockDeviceGlobalImgScopeMem[1], Vals[1]);
}

TEST_F(DeviceGlobalTest, DeviceGlobalImgScopeUseBeforeCopyFrom) {
  REDEFINE_AFTER_TEMPLATED(urEnqueueDeviceGlobalVariableWrite, false);
  REDEFINE_AFTER(urEnqueueDeviceGlobalVariableRead);

  Q.single_task<DeviceGlobalImgScopeTestKernel>([]() {}).wait();

  // Register the cached program as expected for device global memory operation.
  auto CtxImpl = sycl::detail::getSyclObjImpl(Q.get_context());
  sycl::detail::KernelProgramCache::KernelCacheT &KernelCache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  ASSERT_EQ(KernelCache.size(), (size_t)1)
      << "Expect 1 program in kernel cache";
  ExpectedReadWriteURProgram = KernelCache.begin()->first;

  // Expect no write or read yet.
  EXPECT_EQ(DeviceGlobalWriteCounter, 0u);
  EXPECT_EQ(DeviceGlobalReadCounter, 0u);

  int Vals[2] = {42, 1234};
  Q.copy(DeviceGlobalImgScope, Vals).wait();

  EXPECT_EQ(DeviceGlobalWriteCounter, 0u);
  EXPECT_EQ(DeviceGlobalReadCounter, 1u);

  // Check the mocked memory.
  EXPECT_EQ(MockDeviceGlobalImgScopeMem[0], Vals[0]);
  EXPECT_EQ(MockDeviceGlobalImgScopeMem[1], Vals[1]);
}

TEST_F(DeviceGlobalTest, DeviceGlobalImgScopeUseBeforeMemcpyFrom) {
  REDEFINE_AFTER_TEMPLATED(urEnqueueDeviceGlobalVariableWrite, false);
  REDEFINE_AFTER(urEnqueueDeviceGlobalVariableRead);

  Q.single_task<DeviceGlobalImgScopeTestKernel>([]() {}).wait();

  // Register the cached program as expected for device global memory operation.
  auto CtxImpl = sycl::detail::getSyclObjImpl(Q.get_context());
  sycl::detail::KernelProgramCache::KernelCacheT &KernelCache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  ASSERT_EQ(KernelCache.size(), (size_t)1)
      << "Expect 1 program in kernel cache";
  ExpectedReadWriteURProgram = KernelCache.begin()->first;

  // Expect no write or read yet.
  EXPECT_EQ(DeviceGlobalWriteCounter, 0u);
  EXPECT_EQ(DeviceGlobalReadCounter, 0u);

  int Vals[2] = {42, 1234};
  Q.memcpy(Vals, DeviceGlobalImgScope).wait();

  EXPECT_EQ(DeviceGlobalWriteCounter, 0u);
  EXPECT_EQ(DeviceGlobalReadCounter, 1u);

  // Check the mocked memory.
  EXPECT_EQ(MockDeviceGlobalImgScopeMem[0], Vals[0]);
  EXPECT_EQ(MockDeviceGlobalImgScopeMem[1], Vals[1]);
}
