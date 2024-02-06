//==------- test_interop_get_native.cpp - SYCL HIP get_native tests --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <hip/hip_runtime.h>

#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/backend/hip.hpp>

#include "TestGetPlatforms.hpp"

#include <iostream>

using namespace sycl;

struct HipInteropGetNativeTests : public ::testing::TestWithParam<platform> {

protected:
  std::unique_ptr<queue> syclQueue_;
  device syclDevice_;

  void SetUp() override {
    syclDevice_ = GetParam().get_devices()[0];
    syclQueue_ = std::unique_ptr<queue>{new queue{syclDevice_}};
  }

  void TearDown() override { syclQueue_.reset(); }
};

TEST_P(HipInteropGetNativeTests, getNativeDevice) {
  hipDevice_t hipDevice = get_native<backend::ext_oneapi_hip>(syclDevice_);
  char hipDeviceName[2] = {0, 0};
  hipError_t result = hipDeviceGetName(hipDeviceName, 2, hipDevice);
  ASSERT_EQ(result, PI_SUCCESS);
  ASSERT_NE(hipDeviceName[0], 0);
}

TEST_P(HipInteropGetNativeTests, getNativeContext) {
  hipCtx_t hipContext =
      get_native<backend::ext_oneapi_hip>(syclQueue_->get_context());
  ASSERT_NE(hipContext, nullptr);
}

TEST_P(HipInteropGetNativeTests, interopTaskGetMem) {
  buffer<int, 1> syclBuffer(range<1>{1});
  syclQueue_->submit([&](handler &cgh) {
    auto syclAccessor = syclBuffer.get_access<access::mode::read>(cgh);
    cgh.host_task([=](interop_handle ih) {
      hipDeviceptr_t hipPtr =
          ih.get_native_mem<backend::ext_oneapi_hip>(syclAccessor);
      hipDeviceptr_t hipPtrBase;
      size_t hipPtrSize = 0;
      hipCtx_t hipContext =
          get_native<backend::ext_oneapi_hip>(syclQueue_->get_context());
      ASSERT_EQ(PI_SUCCESS, hipCtxPushCurrent(hipContext));
      ASSERT_EQ(PI_SUCCESS,
                hipMemGetAddressRange(&hipPtrBase, &hipPtrSize, hipPtr));
      ASSERT_EQ(PI_SUCCESS, hipCtxPopCurrent(nullptr));
      ASSERT_EQ(sizeof(int), hipPtrSize);
    });
  });
}

TEST_P(HipInteropGetNativeTests, interopTaskGetQueue) {
  hipStream_t hipStream = get_native<backend::ext_oneapi_hip>(*syclQueue_);
  syclQueue_->submit([&](handler &cgh) {
    cgh.host_task([=](interop_handle ih) {
      hipStream_t hipInteropStream =
          ih.get_native_queue<backend::ext_oneapi_hip>();
      ASSERT_EQ(hipInteropStream, hipStream);
    });
  });
}

TEST_P(HipInteropGetNativeTests, hostTaskGetNativeMem) {
  buffer<int, 1> syclBuffer(range<1>{1});
  syclQueue_->submit([&](handler &cgh) {
    auto syclAccessor = syclBuffer.get_access<access::mode::read>(cgh);
    cgh.host_task([=](interop_handle ih) {
      hipDeviceptr_t hipPtr =
          ih.get_native_mem<backend::ext_oneapi_hip>(syclAccessor);
      hipDeviceptr_t hipPtrBase;
      size_t hipPtrSize = 0;
      hipCtx_t hipContext =
          get_native<backend::ext_oneapi_hip>(syclQueue_->get_context());
      ASSERT_EQ(PI_SUCCESS, hipCtxPushCurrent(hipContext));
      ASSERT_EQ(PI_SUCCESS,
                hipMemGetAddressRange(&hipPtrBase, &hipPtrSize, hipPtr));
      ASSERT_EQ(PI_SUCCESS, hipCtxPopCurrent(nullptr));
      ASSERT_EQ(sizeof(int), hipPtrSize);
    });
  });
}

TEST_P(HipInteropGetNativeTests, hostTaskGetNativeQueue) {
  hipStream_t hipStream = get_native<backend::ext_oneapi_hip>(*syclQueue_);
  syclQueue_->submit([&](handler &cgh) {
    cgh.host_task([=](interop_handle ih) {
      hipStream_t hipInteropStream =
          ih.get_native_queue<backend::ext_oneapi_hip>();
      ASSERT_EQ(hipInteropStream, hipStream);
    });
  });
}

TEST_P(HipInteropGetNativeTests, hostTaskGetNativeContext) {
  hipCtx_t hipContext =
      get_native<backend::ext_oneapi_hip>(syclQueue_->get_context());
  syclQueue_->submit([&](handler &cgh) {
    cgh.host_task([=](interop_handle ih) {
      hipCtx_t hipInteropContext =
          ih.get_native_context<backend::ext_oneapi_hip>();
      ASSERT_EQ(hipInteropContext, hipContext);
    });
  });
}

INSTANTIATE_TEST_SUITE_P(
    OnHipPlatform, HipInteropGetNativeTests,
    ::testing::ValuesIn(pi::getPlatformsWithName("HIP BACKEND")));
