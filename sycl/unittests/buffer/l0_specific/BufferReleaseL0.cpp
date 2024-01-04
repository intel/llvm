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

pi_device GlobalDeviceHandle(createDummyHandle<pi_device>());

inline pi_result customMockDevicesGet(pi_platform platform,
                                      pi_device_type device_type,
                                      pi_uint32 num_entries, pi_device *devices,
                                      pi_uint32 *num_devices) {
  if (num_devices)
    *num_devices = 1;

  if (devices && num_entries > 0)
    devices[0] = GlobalDeviceHandle;

  return PI_SUCCESS;
}

inline pi_result customMockContextGetInfo(pi_context context,
                                          pi_context_info param_name,
                                          size_t param_value_size,
                                          void *param_value,
                                          size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_CONTEXT_INFO_NUM_DEVICES: {
    if (param_value)
      *static_cast<pi_uint32 *>(param_value) = 1;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(pi_uint32);
    return PI_SUCCESS;
  }
  case PI_CONTEXT_INFO_DEVICES: {
    if (param_value)
      *static_cast<pi_device *>(param_value) = GlobalDeviceHandle;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(GlobalDeviceHandle);
    break;
  }
  default:;
  }
  return PI_SUCCESS;
}

TEST_F(BufferDestructionCheckL0, BufferWithSizeOnlyInterop) {
  Mock.redefineAfter<sycl::detail::PiApiKind::piContextGetInfo>(
      customMockContextGetInfo);
  Mock.redefineAfter<sycl::detail::PiApiKind::piDevicesGet>(
      customMockDevicesGet);

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