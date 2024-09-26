//==- BufferReleaseL0.cpp --- check delayed destruction of buffer ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../BufferReleaseBase.hpp"

class BufferDestructionCheckL0 : public BufferDestructionCheckCommon<
                                     sycl::backend::ext_oneapi_level_zero> {};

ur_device_handle_t
    GlobalDeviceHandle(mock::createDummyHandle<ur_device_handle_t>());

inline ur_result_t customMockDevicesGet(void *pParams) {
  auto params = *reinterpret_cast<ur_device_get_params_t *>(pParams);
  if (*params.ppNumDevices)
    **params.ppNumDevices = 1;

  if (*params.pphDevices && *params.pNumEntries > 0)
    *params.pphDevices[0] = GlobalDeviceHandle;

  return UR_RESULT_SUCCESS;
}

inline ur_result_t customMockContextGetInfo(void *pParams) {
  auto params = *static_cast<ur_context_get_info_params_t *>(pParams);
  switch (*params.ppropName) {
  case UR_CONTEXT_INFO_NUM_DEVICES: {
    if (*params.ppPropValue)
      *static_cast<uint32_t *>(*params.ppPropValue) = 1;
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(uint32_t);
    return UR_RESULT_SUCCESS;
  }
  case UR_CONTEXT_INFO_DEVICES: {
    if (*params.ppPropValue)
      *static_cast<ur_device_handle_t *>(*params.ppPropValue) =
          GlobalDeviceHandle;
    if (*params.ppPropSizeRet)
      **params.ppPropSizeRet = sizeof(GlobalDeviceHandle);
    break;
  }
  default:;
  }
  return UR_RESULT_SUCCESS;
}

TEST_F(BufferDestructionCheckL0, BufferWithSizeOnlyInterop) {
  mock::getCallbacks().set_after_callback("urContextGetInfo",
                                          &customMockContextGetInfo);
  mock::getCallbacks().set_after_callback("urDeviceGet", &customMockDevicesGet);

  auto Test = [&](sycl::ext::oneapi::level_zero::ownership Ownership) {
    sycl::context ContextForInterop{Plt};
    sycl::queue QueueForInterop =
        sycl::queue{ContextForInterop, sycl::default_selector{}};
    sycl::device DeviceForInterop = QueueForInterop.get_device();

    sycl::backend_traits<sycl::backend::ext_oneapi_level_zero>::return_type<
        sycl::context>
        ZeContext;
    sycl::backend_traits<sycl::backend::ext_oneapi_level_zero>::return_type<
        sycl::device>
        ZeDevice;
    ZeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        ContextForInterop);
    ZeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
        DeviceForInterop);

    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::device>
        InteropDeviceInput{ZeDevice};
    sycl::device InteropDevice =
        sycl::make_device<sycl::backend::ext_oneapi_level_zero>(
            InteropDeviceInput);

    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::context>
        InteropContextInput{
            ZeContext, std::vector<sycl::device>(1, InteropDevice), Ownership};
    sycl::context InteropContext =
        sycl::make_context<sycl::backend::ext_oneapi_level_zero>(
            InteropContextInput);

    sycl::queue Q(InteropContext, sycl::default_selector{});

    MockCmdWithReleaseTracking *MockCmd = NULL;
    {
      using AllocatorTypeTest = sycl::buffer_allocator<int>;
      AllocatorTypeTest allocator;
      sycl::buffer<int, 1, AllocatorTypeTest> Buf(1, allocator);
      MockCmd = addCommandToBuffer(Buf, Q);
      EXPECT_CALL(*MockCmd, Release).Times(1);
    }
    EXPECT_EQ(MockSchedulerPtr->MDeferredMemObjRelease.size(),
              Ownership == sycl::ext::oneapi::level_zero::ownership::transfer);
  };

  Test(sycl::ext::oneapi::level_zero::ownership::keep);
  Test(sycl::ext::oneapi::level_zero::ownership::transfer);
}
